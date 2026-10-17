import json
from pathlib import Path
import tempfile
import unittest

from data import build_tracks, is_complete, render_html, timeline


class TimelineTest(unittest.TestCase):
    def test_labels_offsets_and_unvoiced_gaps(self):
        segment = {'time': [2000, 8000], 'duration': '1 1 1 1 1 1',
                   'note_pitch': '60 62 64 65 0 0', 'note_type': '2 3 2 3 1 1',
                   'text': '我 我 我 你 <SP> <AP>', 'f0': '440 0 880'}
        result = timeline([segment])
        self.assertEqual([n['label'] for n in result['notes']], ['我', '', '我', '你', '', ''])
        self.assertEqual(result['notes'][0]['start'], 2)
        self.assertEqual([n['type'] for n in result['notes']], [2, 3, 2, 3, 1, 1])
        self.assertEqual(result['segments'], [{
            'start': 2, 'end': 8, 'label': '段 1', 'index': '1'
        }])
        self.assertEqual(result['curves'][0], [[2, 69], [2.02, None], [2.04, 81]])

    def test_invalid_parallel_fields(self):
        with self.assertRaises(ValueError):
            timeline([{'time': [0, 1000], 'duration': '1', 'note_pitch': '60 61', 'note_type': '2', 'text': '我'}])

    def test_incomplete_output_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'melody').mkdir()
            (root / 'melody/generated.wav').touch()
            source = root / '中文 空格.wav'
            source.touch()
            calls = []
            def load(path):
                calls.append(path)
                return {'src': 'test', 'duration': 1, 'peaks': []}
            tracks = build_tracks({'directory': tmp, 'source': {'source': str(source)},
                                   'stages': {'melody': {'status': 'running'}}}, load)
            self.assertEqual(calls, [source])
            self.assertIsNone(tracks[1]['src'])

    def test_complete_requires_source_statuses_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'target.wav'
            source.touch()
            for relative in ('preprocess/metadata.json', 'score/generated.wav', 'melody/generated.wav'):
                artifact = root / relative
                artifact.parent.mkdir(exist_ok=True)
                artifact.touch()
            record = {'directory': tmp, 'source': {'source': str(source)}, 'stages': {
                stage: {'status': 'done'} for stage in ('preprocess', 'score', 'melody')
            }}
            self.assertTrue(is_complete(record))
            record['stages']['melody']['status'] = 'running'
            self.assertFalse(is_complete(record))

    def test_metadata_cannot_close_script(self):
        payload = {'lyric': '</script><script>alert(1)</script>'}
        text = render_html('__PAYLOAD__', payload)
        self.assertNotIn('<', text)
        self.assertEqual(json.loads(text), payload)


if __name__ == '__main__':
    unittest.main()
