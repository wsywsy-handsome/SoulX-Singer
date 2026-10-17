#!/usr/bin/env python3
"""Sample recordings, preprocess, and synthesize score/melody variants.
Run with the SoulX-Singer Python environment. Default action is planning only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ratio', type=float, default=0.1, help='Fraction of files, (0, 1]; rounded up')
    p.add_argument('--limit', type=int, help='Optional maximum number of selected files')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--input-dir', type=Path, default=ROOT / 'record_data')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'experiment/batch_targets')
    p.add_argument('--device', default='cuda')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--run', action='store_true', help='Execute; otherwise only inventory and plan')
    p.add_argument('--preprocess-rtf', type=float, default=1.0, help='Assumed compute seconds / audio second')
    p.add_argument('--inference-rtf', type=float, default=1.0, help='Assumed per-mode compute seconds / audio second')
    p.add_argument('--startup-seconds', type=float, default=60, help='Assumed startup per resident worker (2 per batch)')
    args = p.parse_args()
    if not 0 < args.ratio <= 1 or (args.limit is not None and args.limit < 1):
        p.error('ratio must be in (0, 1] and limit must be positive')
    if min(args.preprocess_rtf, args.inference_rtf, args.startup_seconds) < 0:
        p.error('Timing assumptions must be nonnegative')
    import soundfile as sf
    source = args.input_dir.resolve()
    out = args.output_dir.resolve()
    if not source.is_dir():
        p.error(f'Missing input directory: {source}')
    files = sorted(x for x in source.rglob('*') if x.is_file() and x.suffix.lower() in {'.wav', '.mp3', '.flac'})
    if not files:
        p.error('No WAV/MP3/FLAC recordings found')
    random.Random(args.seed).shuffle(files)
    rows = []
    for f in files:
        rel = str(f.relative_to(source))
        rows.append({'source': str(f), 'relative_path': rel,
                     'id': hashlib.sha256(rel.encode()).hexdigest()[:16],
                     'duration_seconds': sf.info(str(f)).duration,
                     'size': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns})
    def selection(ratio):
        return rows[:min(math.ceil(len(rows) * ratio), args.limit or len(rows))]
    def estimate(subset):
        duration = sum(r['duration_seconds'] for r in subset)
        return {'files': len(subset), 'audio_seconds': duration,
                'assumed_wall_seconds': duration * (args.preprocess_rtf + 2 * args.inference_rtf) + 2 * args.startup_seconds}
    selected = selection(args.ratio)
    settings = {'device': args.device, 'fp16': args.fp16, 'language': 'Mandarin',
                'vocal_sep': False, 'max_merge_duration': 30000, 'midi_transcribe': True,
                'prompt_audio': str(ROOT / 'example/audio/zh_prompt.mp3'),
                'prompt_metadata': str(ROOT / 'example/audio/zh_prompt.json'),
                'model': str(ROOT / 'pretrained_models/SoulX-Singer/model.pt'),
                'config': str(ROOT / 'soulxsinger/config/soulxsinger.yaml'),
                'phoneset': str(ROOT / 'soulxsinger/utils/phoneme/phone_set.json'),
                'pitch_shift': 0, 'auto_shift': False}
    # Detect changed model/prompt/config when deciding whether completed stages can be reused.
    for key in ('prompt_audio', 'prompt_metadata', 'model', 'config', 'phoneset'):
        f = Path(settings[key])
        if not f.is_file():
            p.error(f'Missing {key}: {f}')
    settings['asset_stats'] = {key: [Path(settings[key]).stat().st_size, Path(settings[key]).stat().st_mtime_ns]
                               for key in ('prompt_audio', 'prompt_metadata', 'model', 'config', 'phoneset')}
    plan = {'seed': args.seed, 'ratio': args.ratio, 'settings': settings, 'selected': selected,
            'estimate_assumptions': {'preprocess_rtf': args.preprocess_rtf, 'inference_rtf_per_mode': args.inference_rtf,
                                     'startup_seconds_per_process': args.startup_seconds},
            'estimates': {str(r): estimate(selection(r)) for r in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)},
            'selected_estimate': estimate(selected)}
    write_json(out / 'plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('estimates', 'selected_estimate')}, indent=2), flush=True)
    if not args.run:
        print(f'Plan saved: {out / "plan.json"}; add --run to execute. Estimates are assumptions, not benchmarks.')
        return
    jobs = []
    for row in selected:
        directory = out / 'targets' / row['id']
        directory.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha256(json.dumps([row, settings], sort_keys=True).encode()).hexdigest()
        record_path = directory / 'record.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        if record and record.get('signature') != signature:
            raise RuntimeError(f'Inputs/settings changed for {directory}; use a new --output-dir')
        record.update(source=row, settings=settings, signature=signature)
        stages = record.setdefault('stages', {})
        staged = directory / ('input' + Path(row['source']).suffix.lower())
        if not staged.exists():
            staged.symlink_to(row['source'])
        prep = directory / 'preprocess'
        metadata = prep / 'metadata.json'
        commands = [('preprocess', [sys.executable, '-m', 'preprocess.pipeline',
                    '--audio_path', str(staged), '--save_dir', str(prep), '--device', args.device,
                    '--language', 'Mandarin', '--vocal_sep', 'False', '--max_merge_duration', '30000',
                    '--midi_transcribe', 'True'], metadata)]
        for mode in ('score', 'melody'):
            dest = directory / mode
            cmd = [sys.executable, '-m', 'cli.inference', '--device', args.device,
                   '--model_path', settings['model'], '--config', settings['config'],
                   '--phoneset_path', settings['phoneset'], '--prompt_wav_path', settings['prompt_audio'],
                   '--prompt_metadata_path', settings['prompt_metadata'], '--target_metadata_path', str(metadata),
                   '--save_dir', str(dest), '--control', mode, '--pitch_shift', '0']
            if args.fp16:
                cmd.append('--fp16')
            commands.append((mode, cmd, dest / 'generated.wav'))
        write_json(record_path, record)
        for name, cmd, artifact in commands:
            jobs.append({'record_path': str(record_path), 'stage': name,
                         'command': cmd, 'artifact': str(artifact)})
    jobs_path = out / 'jobs.json'
    write_json(jobs_path, jobs)
    workers = {}
    for phase in ('preprocess', 'inference'):
        log = out / f'{phase}_worker.log'
        started = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/batch_targets_worker.py'),
                                     str(jobs_path), phase], cwd=ROOT,
                                    stdout=stream, stderr=subprocess.STDOUT)
        workers[phase] = {'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started,
                          'log': str(log)}
        print(f'{phase} worker finished (exit {result.returncode}); log: {log}', flush=True)
    failures = []
    records = [json.loads((out / 'targets' / r['id'] / 'record.json').read_text()) for r in selected]
    for record in records:
        for stage in ('preprocess', 'score', 'melody'):
            state = record['stages'].get(stage, {})
            if state.get('status') != 'done' or not Path(state.get('artifact', '')).is_file():
                failures.append({'id': record['source']['id'], 'stage': stage, 'state': state})
    completed = [r for r in records if all(r['stages'].get(s, {}).get('status') == 'done' and Path(r['stages'][s]['artifact']).is_file() for s in ('preprocess', 'score', 'melody'))]
    # Legacy per-file timings include model loading; exclude them from resident-worker estimates.
    timed = [r for r in completed if all(r['stages'][s].get('timing_scope') == 'task_only'
                                       for s in ('preprocess', 'score', 'melody'))]
    total_audio = sum(r['source']['duration_seconds'] for r in timed)
    measured_rtf = {s: sum(r['stages'][s]['elapsed_seconds'] for r in timed) / total_audio
                    for s in ('preprocess', 'score', 'melody')} if total_audio else {}
    startup = {}
    for phase in workers:
        timing = out / f'{phase}_startup.json'
        if timing.exists():
            startup[phase] = json.loads(timing.read_text())
    startup_seconds = sum(v['elapsed_seconds'] for v in startup.values())
    write_json(out / 'summary.json', {'records': records, 'failures': failures,
               'workers': workers, 'startup': startup,
               'measured_rtf_excluding_startup': measured_rtf,
               'extrapolated_wall_seconds': {str(r): startup_seconds + sum(measured_rtf.values()) * estimate(selection(r))['audio_seconds']
                                             for r in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)} if total_audio else {}})
    print(f'Finished: {len(completed)}/{len(selected)} complete; summary: {out / "summary.json"}')
    if failures or any(w['returncode'] for w in workers.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
