"""读取 recon 的离线 PT 数据；仅保留基础 CFM 训练所需字段。"""
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

FIELDS = ('waveform', 'phoneme', 'note_pitch', 'note_type', 'mel2note', 'f0')


class SoulXSingerDataset(Dataset):
    def __init__(self, final_list_path, pt_dir, mode='train', seed=24,
                 test_song_count=100, sample_rate=24000, max_sec=16.0):
        if mode not in ('train', 'test'):
            raise ValueError('mode must be train or test')
        songs = {}
        for line in Path(final_list_path).read_text().splitlines():
            if ':' not in line:
                continue
            name, segments = line.split(':', 1)
            segments = [s.strip() for s in segments.split(',') if s.strip()]
            if segments:
                try:
                    segments.sort(key=int)
                except ValueError:
                    segments.sort()
                songs[name.strip()] = segments
        names = sorted(songs)
        if not 0 <= test_song_count < len(names):
            raise ValueError('test_song_count must be smaller than the song count')
        random.Random(seed).shuffle(names)
        selected = names[test_song_count:] if mode == 'train' else names[:test_song_count]
        if not selected:
            raise ValueError(f'Empty {mode} dataset')
        self.songs = {name: songs[name] for name in sorted(selected)}
        self.items = [(name, seg) for name, segs in self.songs.items() for seg in segs]
        self.root, self.mode, self.seed = Path(pt_dir), mode, seed
        self.max_samples = int(max_sec * sample_rate)

    def __len__(self):
        return len(self.items)

    def _load(self, song, segment):
        path = self.root / song / f'{segment}.pt'
        raw = torch.load(path, map_location='cpu', weights_only=True)
        data = {}
        for key in FIELDS:
            value = raw[key]
            # 接受离线一维张量，以及 DataProcessor 输出的 [1, T]。
            if value.ndim == 2 and value.shape[0] == 1:
                value = value.squeeze(0)
            if value.ndim != 1 or not value.numel():
                raise ValueError(f'{path}: {key} must be a nonempty 1-D tensor')
            data[key] = value.float() if key in ('waveform', 'f0') else value.long()
        if data['waveform'].numel() > self.max_samples:
            raise ValueError(f'{path}: waveform exceeds max_sec; filter the index first')
        if data['f0'].numel() != data['mel2note'].numel():
            raise ValueError(f'{path}: f0 and mel2note lengths differ')
        n = data['phoneme'].numel()
        if any(data[k].numel() != n for k in ('note_pitch', 'note_type')):
            raise ValueError(f'{path}: token lengths differ')
        if data['mel2note'].min() < 0 or data['mel2note'].max() >= n:
            raise ValueError(f'{path}: mel2note index out of range')
        return data

    def __getitem__(self, index):
        song, segment = self.items[index]
        rng = random if self.mode == 'train' else random.Random(self.seed + index)
        if self.mode == 'train':
            segments = self.songs[song]
            pos = segments.index(segment)
            candidates = [s for i, s in enumerate(segments) if abs(i - pos) > 1]
            candidates = candidates or [s for s in segments if s != segment] or [segment]
            prompt_song, prompt_segment = song, rng.choice(candidates)
        else:
            others = [s for s in self.songs if s != song]
            prompt_song = rng.choice(others) if others else song
            prompt_segment = rng.choice(self.songs[prompt_song])
        # 数据损坏时明确报错，不沿用源项目可能无限递归的重新采样。
        return {'prompt': self._load(prompt_song, prompt_segment),
                'target': self._load(song, segment)}


def soulx_collate_fn(batch):
    result = {}
    for side in ('prompt', 'target'):
        result[side] = {
            key: pad_sequence([item[side][key] for item in batch], batch_first=True,
                              padding_value=1 if key == 'note_type' else 0)
            for key in FIELDS
        }
        # 必须使用单数键，与 SoulXSinger.forward 的读取接口一致。
        result[f'{side}_mel_len'] = torch.tensor(
            [item[side]['mel2note'].numel() for item in batch], dtype=torch.long)
    return result
