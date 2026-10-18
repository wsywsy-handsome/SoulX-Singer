"""Base CFM training loop, adapted from recon/core/trainer.py."""
import json
import logging
import math
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, FullStateDictConfig,
    FullOptimStateDictConfig, StateDictType,
)

report = logging.getLogger('soulxsinger.train')


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    return data


def cfm_loss(outputs, sigma):
    # 在 FP32 中计算误差与归一化，避免长序列 FP16 求和溢出。
    target = outputs['x'].float() - (1 - sigma) * outputs['noise'].float()
    error = F.l1_loss(outputs['flow_pred'].float(), target, reduction='none')
    mask = outputs['final_mask'].float()
    return (error * mask).sum() / (mask.sum() * error.shape[-1] + 1e-5)


def lr_multiplier(step, warmup_iters, total_iters):
    if step < warmup_iters:
        return step / max(1, warmup_iters)
    progress = min(1.0, (step - warmup_iters) / max(1, total_iters - warmup_iters))
    return 0.5 * (1 + math.cos(math.pi * progress))


def infinite_loader(loader):
    if len(loader) == 0:
        raise ValueError('Training loader is empty; reduce batch_size or disable drop_last')
    epoch = 0
    while True:
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch, epoch
        epoch += 1


def save_checkpoint(model, optimizer, scheduler, scaler, path, step, epoch, config, rank=0):
    # FSDP 模型与优化器状态都必须由所有 rank 参与汇聚。
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        optim_state = optimizer.state_dict()
    if rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # state_dict 可直接由现有 cli/inference.py 加载。
        temporary = path.with_suffix('.tmp')
        torch.save(dict(state_dict=state, optimizer=optim_state,
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        iteration=step, epoch=epoch, config=config), temporary)
        temporary.replace(path)
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def validate(model, loader, device, config, step, rank=0):
    # 与源实现一致，生成一条试听音频，不将其称为全验证集指标。
    # batch_size=1 避免 infer 尚未支持的变长 batch padding。
    import soundfile as sf
    model.eval()
    try:
        batch = move_to_device(next(iter(loader)), device)
        audio = model(batch, run_infer=True, control='melody', use_fp16=False,
                      n_steps=config['infer']['n_steps'], cfg=config['infer']['cfg'])
        if rank == 0:
            path = Path(config['train']['output_dir']) / 'audio'
            path.mkdir(parents=True, exist_ok=True)
            sf.write(path / f'step_{step}.wav', audio.squeeze().float().cpu().numpy(),
                     config['audio']['sample_rate'])
    finally:
        model.train()
        model.mel.eval()
        model.vocoder.eval()


def train(model, loader, optimizer, scheduler, scaler, device, config,
          val_loader=None, rank=0):
    options = config['train']
    model.train()
    model.mel.eval()
    model.vocoder.eval()
    batches = infinite_loader(loader)
    output_dir = Path(options['output_dir'])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, options['total_iters'] + 1):
        batch, epoch = next(batches)
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch, control='random', use_fp16=options['fp16'])
        loss = cfm_loss(outputs, float(config['model']['flow_matching']['sigma']))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if isinstance(model, FSDP):
            model.clip_grad_norm_(options['clip_max_norm'])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), options['clip_max_norm'])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # 溢出跳过参数更新时，不提前推进学习率。
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        value = loss.detach().clone()
        if dist.is_initialized():
            dist.all_reduce(value)
            value /= dist.get_world_size()
        if rank == 0 and (step == 1 or step % options['log_interval'] == 0):
            record = dict(step=step, epoch=epoch, loss=value.item(),
                          lr=optimizer.param_groups[0]['lr'])
            print(record, flush=True)
            report.info('step=%s epoch=%s loss=%.6f lr=%.8g',
                        step, epoch, value.item(), optimizer.param_groups[0]['lr'])
            with (output_dir / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        if step % options['val_iters'] == 0 or step == options['total_iters']:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            output_dir / 'checkpoints' / f'step_{step}.pt',
                            step, epoch, config, rank)
            if rank == 0:
                report.info('Checkpoint saved: %s', output_dir / 'checkpoints' / f'step_{step}.pt')
            if val_loader is not None:
                # 试听不改变训练端 Python/Torch 随机流。
                rng = random.getstate()
                devices = [device.index] if device.type == 'cuda' else []
                try:
                    with torch.random.fork_rng(devices=devices):
                        validate(model, val_loader, device, config, step, rank)
                        if rank == 0:
                            report.info('Validation audio saved: %s', output_dir / 'audio' / f'step_{step}.wav')
                finally:
                    random.setstate(rng)
