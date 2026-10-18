"""Build a song index from actual PT waveform shapes, or benchmark a random sample.

python scripts/build_final_list.py --estimate-only --sample-songs 200
python scripts/build_final_list.py --output data/final_list.txt
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import random
import time

import torch

DEFAULT_ROOT = '/inspire/hdd2/project/multilingualspeechrecognition/chenxie-25019/tieruian/163music_datasets_pt'
FIELDS = ('waveform', 'phoneme', 'note_pitch', 'note_type', 'mel2note', 'f0')


def inspect_song(root, song, max_samples):
    kept, errors = [], []
    count = over = samples = 0
    try:
        paths = sorted(p for p in (root / song).iterdir() if p.suffix == '.pt' and p.is_file())
    except OSError as error:
        return song, kept, count, over, samples, [{'path': str(root / song), 'error': str(error)}]
    for path in paths:
        count += 1
        try:
            # mmap avoids reading audio storage; meta exposes real tensor shapes only.
            try:
                data = torch.load(str(path), map_location='meta', weights_only=True, mmap=True)
            except RuntimeError as error:
                if 'mmap can only be used' not in str(error):
                    raise
                data = torch.load(str(path), map_location='meta', weights_only=True)
            lengths = {}
            for key in FIELDS:
                tensor = data[key]
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f'{key}: not a tensor')
                if tensor.ndim == 2 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.ndim != 1 or tensor.numel() == 0:
                    raise ValueError(f'{key}: invalid shape {tuple(tensor.shape)}')
                lengths[key] = tensor.numel()
            if lengths['f0'] != lengths['mel2note']:
                raise ValueError('f0/mel2note length mismatch')
            if any(lengths[k] != lengths['phoneme'] for k in ('note_pitch', 'note_type')):
                raise ValueError('token length mismatch')
            if any(c in song for c in ':,\n\r') or any(c in path.stem for c in ',\n\r'):
                raise ValueError('name cannot be represented in final_list format')
            if lengths['waveform'] <= max_samples:
                kept.append(path.stem)
                samples += lengths['waveform']
            else:
                over += 1
        except Exception as error:
            errors.append({'path': str(path), 'error': f'{type(error).__name__}: {error}'})
    return song, kept, count, over, samples, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--output', type=Path, default=Path('data/final_list.txt'))
    parser.add_argument('--sample-rate', type=int, default=24000)
    parser.add_argument('--max-sec', type=float, default=16.)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--estimate-only', action='store_true')
    parser.add_argument('--sample-songs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=24)
    args = parser.parse_args()
    if min(args.sample_rate, args.workers, args.sample_songs) <= 0 or not math.isfinite(args.max_sec) or args.max_sec <= 0:
        parser.error('sample-rate, workers, sample-songs and max-sec must be positive')
    torch.set_num_threads(1)
    start = time.monotonic()
    with os.scandir(args.root) as entries:
        songs = sorted(e.name for e in entries if e.is_dir())
    total = len(songs)
    if not total:
        parser.error('no song directories found')
    if args.estimate_only:
        songs = random.Random(args.seed).sample(songs, min(args.sample_songs, total))
    print(f'Song directories: {total}; processing: {len(songs)}; workers: {args.workers}', flush=True)
    partial = args.output.with_suffix(args.output.suffix + '.partial')
    summary_path = args.output.with_suffix('.estimate.json' if args.estimate_only else '.summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out = errors_file = None
    counts = dict(songs_scanned=0, files_scanned=0, songs_kept=0, segments_kept=0, over_limit=0, errors=0, kept_samples=0)
    scan_start = time.monotonic()
    try:
        errors_file = args.output.with_suffix('.estimate.errors.jsonl' if args.estimate_only else '.errors.jsonl').open('w')
        if not args.estimate_only:
            out = partial.open('w')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Bound queued work and memory, even for hundreds of thousands of songs.
            for offset in range(0, len(songs), 128):
                results = pool.map(lambda song: inspect_song(args.root, song, int(args.max_sec * args.sample_rate)), songs[offset:offset + 128])
                for song, kept, count, over, samples, errors in results:
                    counts['songs_scanned'] += 1
                    counts['files_scanned'] += count
                    counts['songs_kept'] += bool(kept)
                    counts['segments_kept'] += len(kept)
                    counts['over_limit'] += over
                    counts['errors'] += len(errors)
                    counts['kept_samples'] += samples
                    if out and kept:
                        out.write(f'{song}:{",".join(kept)}\n')
                    if errors_file:
                        for error in errors:
                            errors_file.write(json.dumps(error, ensure_ascii=False) + '\n')
                elapsed = time.monotonic() - scan_start
                eta = elapsed / counts['songs_scanned'] * (len(songs) - counts['songs_scanned'])
                print(f"{counts['songs_scanned']}/{len(songs)} songs; files={counts['files_scanned']} kept={counts['segments_kept']} errors={counts['errors']} elapsed={elapsed:.1f}s ETA={eta/60:.1f}min", flush=True)
                if out:
                    out.flush()
        if out:
            out.close()
            out = None
            if not counts['segments_kept']:
                raise RuntimeError('No valid segments; existing final index was not replaced')
            if args.output.exists():
                backup = args.output.with_name(args.output.name + f'.backup.{time.time_ns()}')
                import shutil
                shutil.copy2(args.output, backup)
                print(f'Previous index backed up to {backup}', flush=True)
            partial.replace(args.output)
    finally:
        if out:
            out.close()
        if errors_file:
            errors_file.close()
    elapsed = time.monotonic() - scan_start
    summary = dict(root=str(args.root), estimate_only=args.estimate_only, total_songs=total,
                   sample_rate=args.sample_rate, max_sec=args.max_sec, workers=args.workers,
                   **counts, elapsed_seconds=time.monotonic()-start,
                   estimated_total_files=round(counts['files_scanned']/len(songs)*total),
                   estimated_full_scan_hours=elapsed/len(songs)*total/3600,
                   kept_audio_hours=counts['kept_samples']/args.sample_rate/3600)
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Summary: {summary_path}', flush=True)
    if counts['errors']:
        print('Some files/directories were excluded due to errors; inspect the error report before training.', flush=True)


if __name__ == '__main__':
    main()
