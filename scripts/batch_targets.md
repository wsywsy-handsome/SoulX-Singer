# 批量 target 预处理和推理

在项目根目录使用 SoulX-Singer 环境执行：

```bash
# 默认只生成计划；ratio 按文件数量抽样，向上取整，默认 seed=42
/opt/conda/envs/soulxsinger/bin/python scripts/batch_targets.py --ratio 0.1
# 正式执行，同样命令重复运行会跳过已完成的阶段
/opt/conda/envs/soulxsinger/bin/python scripts/batch_targets.py --ratio 0.1 --run
# 先处理 3 条做实测，再增大比例；沿用同一个目录可复用已完成结果
/opt/conda/envs/soulxsinger/bin/python scripts/batch_targets.py --ratio 0.01 --run
```

固定种子打乱排序后的文件列表，不同比例使用同一列表前缀，因此可逐步扩大样本。
支持 WAV/MP3/FLAC。`--limit N` 可限制条数。默认按文件随机抽样，不保证歌手或曲目分层覆盖。
默认 CUDA、全精度、pitch_shift=0，不启用 auto_shift；可加 `--fp16`。
预处理固定 Mandarin、vocal_sep=False、max_merge_duration=30000、midi_transcribe=True。
Prompt 固定为 example/audio/zh_prompt.mp3 和 example/audio/zh_prompt.json；现有推理入口只取 prompt metadata 第一段。

输出目录默认为 experiment/batch_targets，可用 --output-dir 修改。每条录音使用相对路径哈希作为 ID，原路径记录在 JSON 中：

```text
plan.json                      # 选择列表、源音频时长、配置及假设估算
summary.json                   # 本次所选目标的状态、实测耗时及按音频时长外推
 targets/<id>/
   record.json                 # 原路径、参数、命令、阶段状态、耗时、产物路径
   input.wav/mp3/flac          # 指向原录音的符号链接
   input.json                 # pipeline 自动复制的 metadata
   preprocess.log
   score.log
   melody.log
   preprocess/                # metadata.json、vocal.wav、切片、F0 等
   score/generated.wav
   melody/generated.wav
```

执行分为两个常驻工作进程：预处理进程加载一次模型，完成全部待处理录音后退出并释放显存；推理进程加载一次模型，先完成全部 score，再完成全部 melody。两个模式共用同一模型。单条任务失败记录日志并继续，最终返回非零退出码。
工作进程实现位于 scripts/batch_targets_worker.py，由主脚本自动调用。预处理失败的录音不会进入推理。无需处理时不加载模型。
根目录额外保存 jobs.json、preprocess_worker.log、inference_worker.log 和两个 *_startup.json。初始化日志与加载耗时单独记录；每条录音的日志和耗时仅覆盖实际任务。推理仍会逐条处理 prompt 数据，但不重新加载生成模型权重。
重复运行跳过状态为 done 且主产物存在的阶段；失败/中断阶段会重跑。预处理重跑时会使对应的 score/melody 状态失效并重新推理。旧脚本已完成的结果仍可复用，但旧版耗时不参与新执行方式的 RTF 估算。
模型、prompt、配置或源文件大小/修改时间发生变化时拒绝复用该目标，请使用新输出目录。
该检查不包含代码版本及所有预处理模型文件；修改代码或预处理权重后也应使用新目录。
输出路径为绝对路径，整个结果目录移动后不应继续原地断点续跑。

时间估算默认假设预处理 RTF=1、每种推理 RTF=1、每个常驻工作进程启动耗时 60 秒（整个批次共两次）：
`总时间 = 音频秒数 × (预处理RTF + 2 × 单模式推理RTF) + 2 × 启动秒数`。
这不是 GPU 实测或承诺。可用 --preprocess-rtf、--inference-rtf、--startup-seconds 修改假设。
完成少量真实任务后，summary.json 给出不含模型加载的各阶段实测 RTF、单独的加载耗时和各比例外推秒数（加回两次加载耗时）；复用任务的耗时和加载数据可能来自之前的运行；
短音频占比、切片数及 GPU 负载会影响准确性。使用 --limit 时，比例表也会受条数上限约束。
