import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional, Dict, Any, List
from contextlib import nullcontext

from soulxsinger.models.modules.vocoder import Vocoder
from soulxsinger.models.modules.decoder import CFMDecoder
from soulxsinger.models.modules.convnext import ConvNeXtV2Block
from soulxsinger.models.modules.mel_transform import MelSpectrogramEncoder


def _autocast_if(enabled: bool, target_dtype=torch.float16):
    """Return autocast(context) if enabled else no-op context. Use: with _autocast_if(use_amp): ..."""
    return torch.amp.autocast(device_type="cuda", dtype=target_dtype, enabled=True) if enabled else nullcontext()

class SoulXSinger(nn.Module):
    """
    SoulXSinger model.
    """
    def __init__(self, config: Dict):
        super(SoulXSinger, self).__init__()
        audio_cfg = config.audio
        enc_cfg = config.model.encoder
        cfm_cfg = config.model.flow_matching
        
        self.note_text_encoder = nn.Embedding(enc_cfg["vocab_size"], enc_cfg["text_dim"])
        self.note_pitch_encoder = nn.Embedding(256, enc_cfg["pitch_dim"])
        self.note_type_encoder = nn.Embedding(256, enc_cfg["type_dim"])
        self.f0_encoder = nn.Embedding(enc_cfg["f0_bin"], enc_cfg["f0_dim"])

        self.preflow = nn.Sequential(
            *[ConvNeXtV2Block(enc_cfg["text_dim"], enc_cfg["text_dim"] * 2) for _ in range(enc_cfg["num_layers"])]
        )
        self.cfm_decoder = CFMDecoder(cfm_cfg)

        if audio_cfg is None and isinstance(enc_cfg, dict):
            audio_cfg = enc_cfg.get("audio_config")
        self.mel = MelSpectrogramEncoder(audio_cfg)
        self.vocoder = Vocoder()

        # CFM 训练只更新条件编码器和生成网络。
        self.mel.eval()
        self.mel.requires_grad_(False)
        self.vocoder.eval()
        self.vocoder.requires_grad_(False)

    @staticmethod
    def expand_states(h, mel2token):
        """
        Expand the states to the mel-scale.
        args:
            h: states, shape: [B, T, H]
            mel2token: mel2token, shape: [B, F]
        returns:
            h: expanded states, shape: [B, F, H]
        """
        try:
            assert mel2token.max() <= h.size(1) - 1
        except:
            print(f"Warning: mel2token.max() ({mel2token.max()}) is greater than h.size(1) - 1 ({h.size(1) - 1})")
            mel2token = torch.clamp(mel2token, 0, h.size(1)-1)
        mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
        h = torch.gather(h, 1, mel2token_)  # [B, T, H]
        return h

    @staticmethod
    def f0_to_coarse(f0, f0_bin=361, f0_min=32.7031956625, f0_shift=0):
        """
        Convert continuous F0 values to discrete F0 bins (SIL and C1 - B6, 361 bins).
        args:
            f0: continuous F0 values
            f0_bin: number of F0 bins
            f0_min: minimum F0 value
            f0_shift: shift value for F0 bins
        returns:
            f0_coarse: discrete F0 bins
        """
        is_torch = isinstance(f0, torch.Tensor)
        uv_mask = f0 <= 0    

        if is_torch:  
            f0_safe = torch.maximum(f0, torch.tensor(f0_min))
            f0_cents = 1200 * torch.log2(f0_safe / f0_min)
        else:
            f0_safe = np.maximum(f0, f0_min)
            f0_cents = 1200 * np.log2(f0_safe / f0_min)

        f0_coarse = (f0_cents / 20) + 1
        
        if is_torch:
            f0_coarse = torch.round(f0_coarse).long()
            f0_coarse = torch.clamp(f0_coarse, min=1, max=f0_bin - 1)
        else:
            f0_coarse = np.rint(f0_coarse).astype(int)
            f0_coarse = np.clip(f0_coarse, 1, f0_bin - 1)

        f0_coarse[uv_mask] = 0

        if f0_shift != 0:
            if is_torch:
                voiced = f0_coarse > 0
                if voiced.any():
                    shifted = f0_coarse[voiced] + f0_shift
                    f0_coarse[voiced] = torch.clamp(shifted, 1, f0_bin - 1)
            else:
                voiced = f0_coarse > 0
                if np.any(voiced):
                    shifted = f0_coarse[voiced] + f0_shift
                    f0_coarse[voiced] = np.clip(shifted, 1, f0_bin - 1)
        
        return f0_coarse

    def infer(self, meta: dict, auto_shift=False, pitch_shift=0, n_steps=32, cfg=3, control="melody", use_fp16=False, return_mel_only=False):
        
        gt_note_text = meta['target']['phoneme']
        gt_mel2note = meta['target']['mel2note']
        gt_note_type = meta['target']['note_type']

        pt_wav = meta['prompt']['waveform']
        pt_note_text = meta['prompt']['phoneme']
        pt_mel2note = meta['prompt']['mel2note']
        pt_note_type = meta['prompt']['note_type']

        if control == "score":
            gt_note_pitch = meta['target']['note_pitch']
            pt_note_pitch = meta['prompt']['note_pitch']
            gt_f0 = None
            pt_f0 = None
        elif control == "melody":
            gt_f0 = meta['target']['f0']
            pt_f0 = meta['prompt']['f0']
            gt_note_pitch = None
            pt_note_pitch = None
        else:
            raise ValueError(f"Unknown control mode: {control}")

        # calculate auto pitch shift
        if auto_shift and pitch_shift == 0:
            if gt_note_pitch != None and pt_note_pitch != None:
                gt_median = torch.median(gt_note_pitch[gt_note_pitch >= 1])
                pt_median = torch.median(pt_note_pitch[pt_note_pitch >= 1])
                f0_shift = torch.round(pt_median - gt_median).int().item()
            elif gt_f0 != None and pt_f0 != None:
                gt_f0_median = torch.median(gt_f0[gt_f0 > 0])
                pt_f0_median = torch.median(pt_f0[pt_f0 > 0])
                f0_shift = torch.round(torch.log2(pt_f0_median / gt_f0_median) * 1200 / 100).int().item()
            else:
                print("Warning: pitch_shift is True but note_pitch or f0 is None. Set f0_shift to 0.")
                f0_shift = 0
        else:
            f0_shift = pitch_shift

        if gt_f0 is None or pt_f0 is None:
            gt_f0, pt_f0 = torch.zeros_like(gt_mel2note).float().to(gt_mel2note.device), torch.zeros_like(pt_mel2note).float().to(pt_mel2note.device)
        if gt_note_pitch is None or pt_note_pitch is None:
            gt_note_pitch, pt_note_pitch = torch.zeros_like(gt_note_type).int().to(gt_note_type.device), torch.zeros_like(pt_note_type).int().to(pt_note_type.device)

        use_fp16 = use_fp16 and pt_wav.is_cuda
        # mel is kept in fp32 (see build_model: model.mel.float() after model.half())
        pt_mel = self.mel(pt_wav.float() if pt_wav.dtype != torch.float32 else pt_wav)
        # 与模型精度保持一致，同时支持 FP16 和 BF16。
        current_dtype = next(self.parameters()).dtype
        target_dtype = current_dtype if current_dtype in (torch.float16, torch.bfloat16) else torch.float16
        if use_fp16:
            pt_mel = pt_mel.to(target_dtype)
            pt_f0 = pt_f0.to(target_dtype)
            gt_f0 = gt_f0.to(target_dtype)

        len_prompt = pt_note_pitch.shape[1]
        len_prompt_mel = pt_f0.shape[1]

        note_pitch = torch.cat([pt_note_pitch, gt_note_pitch], 1)
        note_text = torch.cat([pt_note_text, gt_note_text], 1)
        note_type = torch.cat([pt_note_type, gt_note_type], 1)
        mel2note = torch.cat([pt_mel2note, gt_mel2note + len_prompt], 1)

        f0_course_pt = self.f0_to_coarse(pt_f0)
        f0_course_gt = self.f0_to_coarse(gt_f0, f0_shift=f0_shift * 5)
        f0_course = torch.cat([f0_course_pt, f0_course_gt], 1)

        note_pitch[note_pitch > 0] = note_pitch[note_pitch > 0] + f0_shift
        note_pitch = torch.clamp(note_pitch, 0, 255)

        with _autocast_if(use_fp16, target_dtype):
            features = self.note_pitch_encoder(note_pitch) + self.note_type_encoder(note_type) + self.note_text_encoder(note_text)

            features = self.preflow(features)
            features = self.expand_states(features, mel2note)
            features = features + self.f0_encoder(f0_course)

            gt_decoder_inp = features[:, len_prompt_mel:, :]
            pt_decoder_inp = features[:, :len_prompt_mel, :]

            generated_mel = self.cfm_decoder.reverse_diffusion(
                pt_mel,
                pt_decoder_inp,
                gt_decoder_inp,
                n_timesteps=n_steps,
                cfg=cfg
            )
            if return_mel_only:
                return generated_mel
            generated_audio = self.vocoder(generated_mel.transpose(1, 2)[0:1, ...]).float()

        return generated_audio

    def forward(self, meta: dict, control="random", use_fp16=False, run_infer=False, return_mel_only=False, **kwargs):
        """
        返回 CFM 速度预测、真实 Mel、噪声及 loss mask；标量 loss 由训练器计算。
        """
        if control == "random":
            control = random.choice(["score", "melody"])

        if run_infer:
            return self.infer(meta=meta, control=control, use_fp16=use_fp16, return_mel_only=return_mel_only, **kwargs)

        gt_wav = meta['target']['waveform']
        gt_note_text = meta['target']['phoneme']
        gt_mel2note = meta['target']['mel2note']
        gt_note_type = meta['target']['note_type']

        pt_wav = meta['prompt']['waveform']
        pt_note_text = meta['prompt']['phoneme']
        pt_mel2note = meta['prompt']['mel2note']
        pt_note_type = meta['prompt']['note_type']

        if control == "score":
            gt_note_pitch = meta['target']['note_pitch']
            pt_note_pitch = meta['prompt']['note_pitch']
            gt_f0 = None
            pt_f0 = None
        elif control == "melody":
            gt_f0 = meta['target']['f0']
            pt_f0 = meta['prompt']['f0']
            gt_note_pitch = None
            pt_note_pitch = None
        elif control == "both":
            gt_note_pitch = meta['target']['note_pitch']
            pt_note_pitch = meta['prompt']['note_pitch']
            gt_f0 = meta['target']['f0']
            pt_f0 = meta['prompt']['f0']
        else:
            raise ValueError(f"Unknown control mode: {control}")

        if gt_f0 is None or pt_f0 is None:
            gt_f0, pt_f0 = torch.zeros_like(gt_mel2note).float().to(gt_mel2note.device), torch.zeros_like(pt_mel2note).float().to(pt_mel2note.device)
        if gt_note_pitch is None or pt_note_pitch is None:
            gt_note_pitch, pt_note_pitch = torch.zeros_like(gt_note_type).int().to(gt_note_type.device), torch.zeros_like(pt_note_type).int().to(pt_note_type.device)

        use_fp16 = use_fp16 and pt_wav.is_cuda
        pt_mel = self.mel(pt_wav.float() if pt_wav.dtype != torch.float32 else pt_wav)
        gt_mel = self.mel(gt_wav.float() if gt_wav.dtype != torch.float32 else gt_wav)

        current_dtype = next(self.parameters()).dtype
        target_dtype = current_dtype if current_dtype in [torch.float16, torch.bfloat16] else torch.float16

        if use_fp16:
            pt_mel = pt_mel.to(target_dtype)
            gt_mel = gt_mel.to(target_dtype)
            pt_f0 = pt_f0.to(target_dtype)
            gt_f0 = gt_f0.to(target_dtype)

        # 两段 Mel 分别对齐后再拼接，避免参考/目标交界错位。
        def align_length(mel_tensor, target_len):
            if mel_tensor.shape[1] > target_len:
                return mel_tensor[:, :target_len, :]
            elif mel_tensor.shape[1] < target_len:
                pad_len = target_len - mel_tensor.shape[1]
                return F.pad(mel_tensor, (0, 0, 0, pad_len))
            return mel_tensor

        pt_mel = align_length(pt_mel, pt_f0.shape[1])
        gt_mel = align_length(gt_mel, gt_f0.shape[1])

        len_prompt = pt_note_pitch.shape[1]
        len_prompt_mel = pt_f0.shape[1]

        note_pitch = torch.cat([pt_note_pitch, gt_note_pitch], 1)
        note_text = torch.cat([pt_note_text, gt_note_text], 1)
        note_type = torch.cat([pt_note_type, gt_note_type], 1)
        mel2note = torch.cat([pt_mel2note, gt_mel2note + len_prompt], 1)

        f0_course_pt = self.f0_to_coarse(pt_f0)
        f0_course_gt = self.f0_to_coarse(gt_f0)
        f0_course = torch.cat([f0_course_pt, f0_course_gt], 1)

        note_pitch = torch.clamp(note_pitch, 0, 255)

        with _autocast_if(use_fp16, target_dtype):
            features = self.note_pitch_encoder(note_pitch) + \
                       self.note_type_encoder(note_type) + \
                       self.note_text_encoder(note_text)
            features = self.preflow(features)
            features = self.expand_states(features, mel2note)
            features = self.f0_encoder(f0_course) + features

            mel = torch.cat([pt_mel, gt_mel], 1)

            B, T_total, _ = features.shape
            # x_mask 标记有效帧；is_prompt 标记保留真实 Mel 的参考帧。
            x_mask = torch.zeros((B, T_total), device=features.device)
            is_prompt = torch.zeros((B, T_total), device=features.device)

            pt_lens = meta.get('prompt_mel_len', None)
            gt_lens = meta.get('target_mel_len', None)

            if pt_lens is not None and gt_lens is not None:
                for i in range(B):
                    p_len = pt_lens[i]
                    t_len = gt_lens[i]
                    x_mask[i, :p_len] = 1.0
                    is_prompt[i, :p_len] = 1.0
                    # 目标从 batch 补齐后的参考段末尾开始，而不是 p_len。
                    start_tgt = len_prompt_mel
                    x_mask[i, start_tgt : start_tgt + t_len] = 1.0
            else:
                x_mask = torch.ones((B, T_total), device=features.device)
                is_prompt[:, :len_prompt_mel] = 1.0

            decoder_inp = features

            noise, x, flow_pred, final_mask, prompt_len = self.cfm_decoder(
                mel, x_mask, decoder_inp, is_prompt
            )

        return {
            "flow_pred": flow_pred,
            "noise": noise,
            "x": x,
            "final_mask": final_mask
        }
