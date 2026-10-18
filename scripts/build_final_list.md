# 生成正式训练索引

在项目根目录、soulxsinger 环境执行：

```bash
python scripts/build_final_list.py --output data/final_list.txt --workers 8
```

默认扫描 `/inspire/hdd2/project/multilingualspeechrecognition/chenxie-25019/tieruian/163music_datasets_pt`
下每个歌曲目录的 `.pt`。按实际 `waveform` 形状（支持 `[L]`、`[1,L]`）筛选
`0 < L <= 16 * 24000`，不以文件名时间戳为准。输出为 `歌曲名:片段ID,...`，
只包含至少一个合格片段的歌曲。采样率假设为 24 kHz，不重采样。

内存映射读取 Tensor 元数据，不扫描波形数值；检查六个必需字段、非空一维形状、
F0/mel2note 长度及 token 长度。不验证 NaN、音素范围或 mel2note 索引值，
所以不能替代全量数据质量审计。兼容旧 PT 格式的回退读取可能更慢。

每 128 首歌输出一次进度和 ETA。完成后才用 `.partial` 原子替换最终文件；
已有索引先备份为 `.backup.<时间戳>`。中断时最终索引保持原样；重跑会从头扫描。
不要同时运行多个写入同一输出路径的任务。

旁路文件：
- `final_list.summary.json`：扫描、保留、超长、错误数量及耗时。
- `final_list.errors.jsonl`：无法读取或字段/形状异常的数据，跳过但不删除源文件。

先实测估算（不替换最终索引）：

```bash
python scripts/build_final_list.py --estimate-only --sample-songs 1000 \
  --seed 107 --workers 8 --output outputs/dataset_audit/final_list.txt
```

估算采用随机抽取的歌曲目录，按总目录数外推，结果写入 `.estimate.json`。
磁盘缓存、共享存储负载和 PT 复杂度会影响耗时，ETA 不是保证。
正式训练前检查错误报告，并自行设置合适的 `test_song_count`（当前小规模测试配置为 5）。
