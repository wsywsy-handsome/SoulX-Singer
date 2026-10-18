"""python -m cli.train --config ...; torchrun --nproc_per_node=2 -m cli.train ..."""
import argparse
import functools
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from soulxsinger.training.dataset import SoulXSingerDataset, soulx_collate_fn
from soulxsinger.training.trainer import lr_multiplier, train
from soulxsinger.training.logging_utils import training_logs


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description='SoulX-Singer base CFM training')
    parser.add_argument('--config', default='soulxsinger/config/train.yaml')
    parser.add_argument('--checkpoint', help='Initial model weights (not optimizer resume)')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    rank = int(os.environ.get('RANK', 0))
    with training_logs(cfg.train.output_dir, rank) as report:
        report.info('Config=%s checkpoint=%s', args.config,
                    args.checkpoint or cfg.train.get('checkpoint'))
        run_training(args, cfg)


def run_training(args, cfg):
    report = logging.getLogger('soulxsinger.train')
    config = OmegaConf.to_container(cfg, resolve=True)
    options = config['train']
    for key in ('total_iters', 'val_iters', 'log_interval', 'batch_size'):
        if options[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= options['warmup_iters'] < options['total_iters']:
        raise ValueError('Require 0 <= warmup_iters < total_iters')
    checkpoint_path = args.checkpoint or options.get('checkpoint')
    if not checkpoint_path:
        raise ValueError('Provide --checkpoint including pretrained vocoder weights')
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if world_size > 1 and device.type != 'cuda':
        raise ValueError('FSDP multi-process training requires CUDA')
    if options['fp16'] and device.type != 'cuda':
        raise ValueError('fp16 requires CUDA; set train.fp16=false for CPU')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('nccl')
    try:
        seed = options['seed'] + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        data = config['data']
        shared = dict(final_list_path=data['final_list_path'], pt_dir=data['pt_dir'],
                      seed=options['seed'], test_song_count=data['test_song_count'],
                      sample_rate=config['audio']['sample_rate'], max_sec=data['max_sec'])
        dataset = SoulXSingerDataset(**shared, mode='train')
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     seed=options['seed']) if world_size > 1 else None
        loader = DataLoader(dataset, batch_size=options['batch_size'], sampler=sampler,
                            shuffle=sampler is None, collate_fn=soulx_collate_fn,
                            num_workers=options['num_workers'], worker_init_fn=seed_worker,
                            pin_memory=device.type == 'cuda', drop_last=True)
        if not len(loader):
            raise ValueError('Not enough training samples for one batch')
        # 所有 rank 使用相同的单条验证数据，FSDP 推理调用次数保持一致。
        val_loader = None
        if data['test_song_count']:
            val_loader = DataLoader(SoulXSingerDataset(**shared, mode='test'),
                                    batch_size=1, collate_fn=soulx_collate_fn)
        report.info('Data ready: train_samples=%s validation_samples=%s batch_size=%s world_size=%s',
                    len(dataset), len(val_loader.dataset) if val_loader else 0,
                    options['batch_size'], world_size)
        from soulxsinger.models.soulxsinger import SoulXSinger
        from soulxsinger.models.modules.llama import LlamaNARDecoderLayer
        model = SoulXSinger(cfg).to(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint.get('weights', checkpoint))
        model.load_state_dict(state, strict=True)
        report.info('Model weights loaded: %s; device=%s', checkpoint_path, device)
        del checkpoint, state
        if world_size > 1:
            policy = functools.partial(transformer_auto_wrap_policy,
                                       transformer_layer_cls={LlamaNARDecoderLayer})
            # 参数保留 FP32；前向内部按 fp16 开关 autocast，Mel 提取保持 FP32。
            model = FSDP(model, auto_wrap_policy=policy, device_id=device,
                         use_orig_params=True, sync_module_states=True)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=options['lr'], betas=(0.9, 0.95), eps=1e-6, weight_decay=0.1)
        scheduler = LambdaLR(optimizer, functools.partial(
            lr_multiplier, warmup_iters=options['warmup_iters'],
            total_iters=options['total_iters']))
        scaler_cls = ShardedGradScaler if world_size > 1 else torch.cuda.amp.GradScaler
        scaler = scaler_cls(enabled=options['fp16'])
        train(model, loader, optimizer, scheduler, scaler, device, config, val_loader, rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
