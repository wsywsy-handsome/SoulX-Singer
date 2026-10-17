"""Read-only adapters for batch target artifacts; no model imports."""
import base64
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def discover(root):
    records = []
    for path in sorted((root / 'targets').glob('*/record.json')):
        try:
            record = read_json(path)
            record['directory'] = str(path.parent)
            records.append(record)
        except (OSError, ValueError):
            continue
    return records


def is_complete(record):
    directory = Path(record['directory'])
    artifacts = {
        'preprocess': directory / 'preprocess/metadata.json',
        'score': directory / 'score/generated.wav',
        'melody': directory / 'melody/generated.wav',
    }
    stages = record.get('stages', {})
    return Path(record.get('source', {}).get('source', '')).is_file() and all(
        stages.get(stage, {}).get('status') == 'done' and path.is_file()
        for stage, path in artifacts.items()
    )


def timeline(segments, f0_step=0.02):
    """Preserve absolute segment starts and native 480/24000-second F0 grid."""
    notes, curves, segment_ranges, warnings = [], [], [], []
    previous = None
    end = 0.0
    for segment_number, segment in enumerate(sorted(segments, key=lambda x: x['time'][0]), start=1):
        start, stop = (float(t) / 1000 for t in segment['time'])
        segment_ranges.append({
            'start': start,
            'end': stop,
            'label': f'段 {segment_number}',
            'index': str(segment.get('index', segment_number)),
        })
        end = max(end, stop)
        fields = [str(segment[k]).split() for k in ('duration', 'note_pitch', 'note_type', 'text')]
        if len({len(f) for f in fields}) != 1:
            raise ValueError(f"{segment.get('index', '?')}: duration / note_pitch / note_type / text 长度不一致")
        cursor = start
        for dur, pitch, kind, lyric in zip(*fields):
            dur, pitch, kind = float(dur), int(pitch), int(kind)
            if kind not in (1, 2, 3) or not math.isfinite(dur) or dur < 0:
                raise ValueError(f'无效音符：duration={dur}, type={kind}；note type 应为 1、2 或 3')
            label = '' if lyric in ('<SP>', '<AP>') or (kind == 3 and lyric == previous) else lyric
            notes.append(dict(start=cursor, end=cursor + dur, pitch=pitch, type=kind,
                              lyric=lyric, label=label))
            previous = lyric
            cursor += dur
        end = max(end, cursor)
        if abs(cursor - stop) > 0.15:
            warnings.append(f"{segment.get('index', '?')}: 音符累计时长与片段边界相差 {cursor-stop:+.2f}s（保留原始时长）")
        points = []
        for i, token in enumerate(str(segment.get('f0', '')).split()):
            t = start + i * f0_step
            if t > stop:
                break
            hz = float(token)
            points.append([t, 69 + 12 * math.log2(hz / 440) if hz > 0 and math.isfinite(hz) else None])
        curves.append(points)
    return dict(notes=notes, curves=curves, segments=segment_ranges,
                duration=end, warnings=warnings)


def audio_payload(path):
    path = Path(path)
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    mono = audio.mean(axis=1)
    # Fixed-size waveform overview; retain actual duration for each track.
    bins = min(1400, len(mono))
    edges = np.linspace(0, len(mono), bins + 1, dtype=int)
    peaks = [float(np.max(np.abs(mono[a:b]))) for a, b in zip(edges[:-1], edges[1:])]
    mime = {'.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.flac': 'audio/flac'}.get(path.suffix.lower(), 'audio/wav')
    return {'src': f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode(),
            'duration': len(mono) / sr, 'peaks': peaks}


def build_tracks(record, loader=audio_payload):
    directory = Path(record['directory'])
    tracks = []
    for key, name, path in [('target', 'Target', Path(record['source']['source'])),
                             ('melody', 'Output · Melody', directory / 'melody/generated.wav'),
                             ('score', 'Output · Score', directory / 'score/generated.wav')]:
        status = record.get('stages', {}).get(key, {}).get('status', 'pending')
        ready = path.is_file() and (key == 'target' or status == 'done')
        track = dict(key=key, name=name, status='ready' if ready else status, src=None)
        if ready:
            try:
                track.update(loader(path))
            except (OSError, RuntimeError, ValueError) as e:
                track.update(status='读取失败', error=str(e), src=None)
        tracks.append(track)
    return tracks


def render_html(template, payload):
    # Metadata is data, never executable HTML (lyrics and filenames may contain markup).
    return template.replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c'))
