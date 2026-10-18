"""Exercise the real model control paths with lightweight acoustic modules."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from soulxsinger.models.soulxsinger import SoulXSinger


class Mel(nn.Module):
    def forward(self, wav):
        return wav.reshape(wav.shape[0], -1, 480).mean(-1)[..., None].expand(-1, -1, 4)


class Decoder(nn.Module):
    def forward(self, mel, mask, features, prompt):
        noise = torch.randn_like(mel)
        return noise, mel, features + noise, (mask * (1 - prompt))[..., None], None

    def reverse_diffusion(self, mel, prompt, target, **kwargs):
        return target + prompt.mean(1, keepdim=True) + mel.mean(1, keepdim=True) + torch.randn_like(target)


def model_and_batch():
    model = SoulXSinger.__new__(SoulXSinger)
    nn.Module.__init__(model)
    model.note_text_encoder = nn.Embedding(30, 4)
    model.note_pitch_encoder = nn.Embedding(256, 4)
    model.note_type_encoder = nn.Embedding(256, 4)
    model.f0_encoder = nn.Embedding(361, 4)
    model.preflow = nn.Identity()
    model.mel = Mel()
    model.cfm_decoder = Decoder()
    batch = {}
    for side in ('prompt', 'target'):
        batch[side] = dict(waveform=torch.randn(1, 1440), phoneme=torch.tensor([[1, 2]]),
                           note_type=torch.tensor([[1, 2]]), note_pitch=torch.tensor([[60, 65]]),
                           f0=torch.full((1, 3), 220.), mel2note=torch.tensor([[0, 1, 1]]))
    return model, batch


class NeitherTests(unittest.TestCase):
    def test_neither_ignores_pitch_in_both_paths(self):
        model, batch = model_and_batch()
        changed = copy.deepcopy(batch)
        for side in changed:
            changed[side]['f0'].fill_(880)
            changed[side]['note_pitch'].fill_(90)
        seen = []
        hooks = [module.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
                 for module in (model.note_pitch_encoder, model.f0_encoder)]
        for infer in (False, True):
            def run(data):
                torch.manual_seed(24)
                return model(data, control='neither', run_infer=infer, return_mel_only=True)
            first, second = run(batch), run(changed)
            if infer:
                torch.testing.assert_close(first, second)
            else:
                for key in first:
                    torch.testing.assert_close(first[key], second[key])
                first['flow_pred'].sum().backward()
                self.assertGreater(model.note_pitch_encoder.weight.grad[0].abs().sum(), 0)
            missing = copy.deepcopy(batch)
            for side in missing:
                del missing[side]['f0'], missing[side]['note_pitch']
            run(missing)  # model itself must not read these fields
        self.assertTrue(all(torch.count_nonzero(x) == 0 for x in seen))
        for hook in hooks:
            hook.remove()
        changed['target']['phoneme'].fill_(3)
        torch.manual_seed(24)
        a = model.infer(batch, control='neither', return_mel_only=True)
        torch.manual_seed(24)
        b = model.infer(changed, control='neither', return_mel_only=True)
        self.assertFalse(torch.equal(a, b))

    def test_existing_controls_and_rejections(self):
        model, batch = model_and_batch()
        for control in ('score', 'melody'):
            with patch('soulxsinger.models.soulxsinger.random.choice', return_value=control) as choice:
                torch.manual_seed(1)
                a = model(batch, control='random')
                choice.assert_called_once_with(['score', 'melody'])
            torch.manual_seed(1)
            b = model(batch, control=control)
            torch.testing.assert_close(a['flow_pred'], b['flow_pred'])
            model.infer(batch, control=control, return_mel_only=True)
        model(batch, control='both')
        for kwargs in ({'auto_shift': True}, {'pitch_shift': 1}, {'pitch_shift': -1}):
            with self.assertRaisesRegex(ValueError, 'neither'):
                model.infer(batch, control='neither', **kwargs)
        for infer in (False, True):
            with self.assertRaisesRegex(ValueError, 'Unknown control'):
                model(batch, control='invalid', run_infer=infer)
