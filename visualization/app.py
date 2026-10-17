"""streamlit run visualization/app.py"""
import os
from pathlib import Path
import sys

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import audio_payload, build_tracks, discover, is_complete, read_json, render_html, timeline

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='SoulX · Listening Room', page_icon='🎹', layout='wide')
st.markdown('''<style>
.stApp{background:#0d1119;color:#e6edf5} .block-container{padding-top:2rem;max-width:1600px}
h1{letter-spacing:-1.5px} [data-testid="stSidebar"]{background:#121925}
</style>''', unsafe_allow_html=True)
st.caption('SOULX SINGER  /  LISTENING ROOM')
st.title('听见差异，看见旋律')
st.caption('同一条 target 的三路试听 · MIDI、歌词与 F0 共用绝对时间轴')
with st.sidebar:
    st.header('录音库')
    root = Path(st.text_input('批处理结果目录', os.environ.get('SOULX_BATCH_DIR', str(ROOT / 'experiment/batch_targets')))).expanduser()
    st.button('↻ 刷新结果', help='读取新完成的输出；刷新会重置当前播放器。')
    search = st.text_input('搜索歌手 / 曲目 / 文件名')
    f0_step = st.number_input('F0 帧间隔（毫秒）', min_value=1.0, max_value=100.0, value=20.0,
                             help='当前预处理默认 hop_size=480、sample_rate=24000，即 20 ms。')
    st.caption('只读取结果文件，不加载模型、不占用 GPU。')
records = discover(root)
if not records:
    st.info('尚无可读取的 record.json。请确认批处理结果目录。')
    st.stop()

@st.fragment(run_every='10s')
def status():
    latest = discover(root)
    counts = [sum(r.get('stages', {}).get(k, {}).get('status') == 'done' for r in latest)
              for k in ('preprocess', 'melody', 'score')]
    ready = sum(is_complete(r) for r in latest)
    cols = st.columns(5)
    for c, title, value in zip(cols, ['批次录音', '已对齐', 'Melody 完成', 'Score 完成', '可试听'], [len(latest), *counts, ready]):
        c.metric(title, value)
status()
complete = [r for r in records if is_complete(r)]
filtered = [r for r in complete if search.lower() in r['source']['relative_path'].lower()]
if not filtered:
    st.info('没有匹配且已完成预处理、Melody 和 Score 推理的录音。')
    st.stop()
choices = {r['source']['id']: r for r in filtered}
selected = st.selectbox('选择 target', list(choices), format_func=lambda k: choices[k]['source']['relative_path'])
record = choices[selected]
metadata = Path(record['directory']) / 'preprocess/metadata.json'
if record.get('stages', {}).get('preprocess', {}).get('status') != 'done' or not metadata.is_file():
    st.info('该录音仍在预处理，完成后点击“刷新结果”。')
    st.stop()

@st.cache_data(max_entries=12, show_spinner=False)
def cached_audio(path, mtime_ns, size):
    return audio_payload(Path(path))

def load_audio(path):
    stat = path.stat()
    return cached_audio(str(path), stat.st_mtime_ns, stat.st_size)

try:
    with st.spinner('正在读取音频与对齐数据…'):
        payload = timeline(read_json(metadata), f0_step=f0_step / 1000)
        payload['tracks'] = build_tracks(record, load_audio)
        payload['duration'] = max(payload['duration'], *(t.get('duration', 0) for t in payload['tracks']))
    template = Path(__file__).with_name('player.html').read_text()
    html = render_html(template, payload)
    if hasattr(st, 'iframe'):
        st.iframe(html, height=980)
    else:
        components.html(html, height=980, scrolling=True)
    for warning in payload['warnings']:
        st.warning(warning)
    with st.expander('数据与显示规则'):
        st.write('Type 1：休止 / 间隔；Type 2：起始音；Type 3：延长音。起始音始终标词；延长音与前一个音同词时省略。<SP>、<AP> 不标词。')
        st.write('所有模式显示同一份 target MIDI / F0。F0=0 的无声区间断开；MIDI pitch=0 的休止音显示在底部独立轨道。')
        st.write('上方计数每 10 秒刷新，不打断试听。新输出完成后点击侧栏“刷新结果”加载。切换音源保留时间位置；若新音源更短，则停在其结尾。')
        st.code(str(metadata))
except (OSError, ValueError, KeyError, RuntimeError) as e:
    st.error(f'暂时无法读取该录音：{e}')
