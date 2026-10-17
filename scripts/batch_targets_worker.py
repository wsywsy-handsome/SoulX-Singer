"""Internal resident workers. Invoke via batch_targets.py."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import sys
import time
import traceback

from batch_targets import ROOT, write_json


def done(job):
    record = json.loads(Path(job['record_path']).read_text())
    return (record['stages'].get(job['stage'], {}).get('status') == 'done'
            and Path(job['artifact']).is_file())


def run_worker(jobs_path, phase):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    jobs = json.loads(jobs_path.read_text())
    jobs = [j for j in jobs if (j['stage'] == 'preprocess') == (phase == 'preprocess')]
    # Process every score task before any melody task, sharing the same model.
    jobs.sort(key=lambda j: {'preprocess': 0, 'score': 1, 'melody': 2}[j['stage']])
    pending = []
    for job in jobs:
        if done(job):
            continue
        record = json.loads(Path(job['record_path']).read_text())
        if phase == 'inference':
            prep = record['stages'].get('preprocess', {})
            if prep.get('status') != 'done' or not Path(prep.get('artifact', '')).is_file():
                continue
        pending.append(job)
    if not pending:
        print('No pending tasks; no models loaded.', flush=True)
        return
    settings = json.loads(Path(pending[0]['record_path']).read_text())['settings']
    started = time.monotonic()
    if phase == 'preprocess':
        from preprocess.pipeline import PreprocessPipeline
        pipeline = PreprocessPipeline(device=settings['device'], language='Mandarin',
                                      save_dir=str(Path(pending[0]['artifact']).parent),
                                      vocal_sep=False, max_merge_duration=30000, midi_transcribe=True)
    else:
        from cli.inference import build_model, process
        from soulxsinger.utils.file_utils import load_config
        config = load_config(settings['config'])
        model = build_model(settings['model'], config, device=settings['device'], use_fp16=settings['fp16'])
    write_json(jobs_path.parent / f'{phase}_startup.json',
               {'elapsed_seconds': time.monotonic() - started, 'timing_scope': 'imports_and_model_load'})
    failures = 0
    for job in pending:
        path = Path(job['record_path'])
        record = json.loads(path.read_text())
        name = job['stage']
        directory = path.parent
        artifact = Path(job['artifact'])
        log = directory / f'{name}.log'
        state = {'status': 'running', 'command': job['command'], 'execution': 'resident_worker',
                 'log': str(log), 'artifact': str(artifact), 'timing_scope': 'task_only'}
        record['stages'][name] = state
        if name == 'preprocess':
            # Rebuilt metadata invalidates downstream results even if old WAVs remain.
            for mode in ('score', 'melody'):
                record['stages'].pop(mode, None)
        write_json(path, record)
        print(f'{record["source"]["relative_path"]}: {name}', flush=True)
        started = time.monotonic()
        with log.open('w') as stream, redirect_stdout(stream), redirect_stderr(stream):
            try:
                if name == 'preprocess':
                    pipeline.save_dir = str(artifact.parent)
                    pipeline.vocal_detector.cut_wavs_output_dir = str(artifact.parent / 'cut_wavs')
                    staged = directory / ('input' + Path(record['source']['source']).suffix.lower())
                    pipeline.run(str(staged))
                    segments = json.loads(artifact.read_text())
                    if not segments:
                        raise RuntimeError('Preprocessing produced no segments')
                    record['segment_count'] = len(segments)
                    record['aligned_seconds'] = sum((x['time'][1] - x['time'][0]) / 1000 for x in segments)
                else:
                    args = argparse.Namespace(device=settings['device'], use_fp16=settings['fp16'],
                        phoneset_path=settings['phoneset'], prompt_wav_path=settings['prompt_audio'],
                        prompt_metadata_path=settings['prompt_metadata'],
                        target_metadata_path=str(directory / 'preprocess/metadata.json'),
                        save_dir=str(artifact.parent), control=name,
                        auto_shift=settings['auto_shift'], pitch_shift=settings['pitch_shift'])
                    process(args, config, model)
                if not artifact.is_file():
                    raise RuntimeError(f'Missing artifact: {artifact}')
                state['status'] = 'done'
            except Exception as e:
                traceback.print_exc()
                state.update(status='failed', error=str(e))
                failures += 1
            finally:
                state['elapsed_seconds'] = time.monotonic() - started
                write_json(path, record)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs', type=Path)
    parser.add_argument('phase', choices=['preprocess', 'inference'])
    args = parser.parse_args()
    run_worker(args.jobs, args.phase)
