# SoulX Listening Room

从项目根目录启动（当前 soulxsinger 环境已有 Streamlit）：

```bash
/opt/conda/envs/soulxsinger/bin/python -m streamlit run visualization/app.py --server.address 127.0.0.1 --server.port 8501
```

远程机器请通过 IDE 转发 8501 端口，或使用 SSH 本地端口转发后在本机打开 http://localhost:8501。
其他环境可安装 `visualization/requirements.txt`。不需要 CUDA，不导入模型，也不修改批处理数据。

默认读取 `experiment/batch_targets`，可在侧栏修改目录，或设置 `SOULX_BATCH_DIR`。
下拉框只显示预处理、Melody 和 Score 推理均完成且产物存在的录音。选择录音后，使用 Target / Melody / Score 三段切换器选择当前音源；三张卡片同时展示波形和就绪状态。
当前只有一个播放器，切换会保留秒数及播放状态；音源更短则停在末尾。
点击波形、卷帘或拖动进度条定位；支持变速、时间缩放、横向平移及自动跟随播放头。
音符悬停显示歌词、MIDI 音高、type 和时间。

- metadata 和界面统一使用项目定义：1=休止/间隔、2=起始、3=延长。
- 起始音始终标词；延长音与前一个音同词则不标；`<SP>` / `<AP>` 不标。不同词的延长音保留标签。
- 所有试听模式展示 target MIDI 与 F0，不暗示这是生成结果重新提取的音高。
- 卷帘使用浅蓝交替底色区分 preprocess 片段，并在每个片段的起止位置绘制蓝色虚线；片段起点标注“段 N”。边界直接取 metadata 的 `time` 字段。
- 时间轴保留 segment 的 `time[0]/1000` 起点，音符按原始 duration 累加，不将各 segment 首尾拼接。
- F0 默认 20ms 一帧（预处理 480/24000），转换为连续 MIDI 音高；0 值断开曲线。可在侧栏调整帧间隔。
- MIDI pitch=0 的音符显示在 REST 轨道；有音高的 type 1 仍按实际音高绘制。
- 播放头由 HTMLAudioElement.currentTime + requestAnimationFrame 更新；暂停/拖动/变速沿用同一个音频时钟。
- 计数每 10 秒局部更新，不打断音频（[Streamlit fragments](https://docs.streamlit.io/develop/concepts/architecture/fragments)）。新产物需手动刷新加载；手动刷新会重置播放。
- 仅当 record.json 标记该模式 done 且音频存在时加载输出，避免读取推理正在写入的 WAV。
- 音频以 data URL 送入浏览器，适合本项目的短录音；长录音会占用较多内存和传输量。

验证：`python -m unittest discover -s visualization -p 'test_*.py'`。
