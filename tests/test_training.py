import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader

from soulxsinger.training.dataset import SoulXSingerDataset, soulx_collate_fn
from soulxsinger.training.trainer import cfm_loss, lr_multiplier, train


def segment(frames):
    return dict(waveform=torch.randn(frames * 480), f0=torch.full((frames,), 220.),
                mel2note=torch.zeros(frames, dtype=torch.long),
                phoneme=torch.tensor([1]), note_pitch=torch.tensor([60]),
                note_type=torch.tensor([2]))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.mel, self.vocoder = nn.Identity(), nn.Identity()

    def forward(self, batch, run_infer=False, **kwargs):
        if run_infer:
            return torch.zeros(1, 960)
        target = batch['target']['f0'][..., None] / 220
        mask = torch.arange(target.shape[1])[None, :] < batch['target_mel_len'][:, None]
        return dict(flow_pred=self.weight.expand_as(target), x=target,
                    noise=torch.zeros_like(target), final_mask=mask[..., None])


class TrainingTests(unittest.TestCase):
    def test_loss_masks_and_sigma(self):
        pred = torch.tensor([[[2., 2.], [999., 999.]]], requires_grad=True)
        out = dict(flow_pred=pred, x=torch.full_like(pred, 3.),
                   noise=torch.full_like(pred, 2.), final_mask=torch.tensor([[[1.], [0.]]]))
        loss = cfm_loss(out, 0.5)  # target = 3 - 0.5 * 2 = 2
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(pred.grad[0, 1].abs().sum().item(), 0.)

    def test_collate_lengths(self):
        batch = soulx_collate_fn([
            dict(prompt=segment(3), target=segment(2)),
            dict(prompt=segment(5), target=segment(4))])
        self.assertEqual(batch['prompt']['f0'].shape, (2, 5))
        self.assertEqual(batch['prompt_mel_len'].tolist(), [3, 5])
        self.assertEqual(batch['target_mel_len'].tolist(), [2, 4])

    def test_dataset_split_and_validation_pairing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'index.txt').write_text('a:0,1,2\nb:0,1,2\nc:0,1,2\n')
            for song in ('a', 'b', 'c'):
                (root / song).mkdir()
                for index in range(3):
                    torch.save(segment(index + 2), root / song / f'{index}.pt')
            args = dict(final_list_path=root/'index.txt', pt_dir=root, test_song_count=1)
            training = SoulXSingerDataset(**args)
            validation = SoulXSingerDataset(**args, mode='test')
            self.assertFalse(set(training.songs) & set(validation.songs))
            self.assertTrue(torch.equal(validation[0]['prompt']['waveform'],
                                        validation[0]['prompt']['waveform']))
            soulx_collate_fn([training[0], training[1]])

    def test_train_save_and_validate(self):
        with tempfile.TemporaryDirectory() as folder:
            model = TinyModel()
            loader = DataLoader([dict(prompt=segment(3), target=segment(4))],
                                collate_fn=soulx_collate_fn)
            optim = AdamW(model.parameters(), lr=0.1)
            scheduler = LambdaLR(optim, lambda step: lr_multiplier(step, 0, 2))
            scaler = ShardedGradScaler(enabled=False)
            cfg = dict(train=dict(output_dir=folder, total_iters=2, log_interval=1,
                                  val_iters=1, fp16=False, clip_max_norm=1.),
                       model=dict(flow_matching=dict(sigma=1e-5)),
                       audio=dict(sample_rate=24000), infer=dict(n_steps=1, cfg=0))
            train(model, loader, optim, scheduler, scaler, torch.device('cpu'), cfg, loader)
            self.assertGreater(model.weight.item(), 0)
            state = torch.load(Path(folder)/'checkpoints/step_2.pt', weights_only=False)
            clone = TinyModel()
            clone.load_state_dict(state['state_dict'], strict=True)
            self.assertEqual(clone.weight.item(), model.weight.item())
            self.assertTrue((Path(folder)/'audio/step_2.wav').exists())
            self.assertTrue(model.training)
            self.assertFalse(model.vocoder.training)
            self.assertEqual(state['iteration'], 2)


if __name__ == '__main__':
    unittest.main()
