# 基础 CFM 训练

从 `recon/main.py`、`core/trainer.py`、`core/test.py` 和基础 PT 数据加载逻辑提取，接入本项目的 `SoulXSinger.forward()`。不包含 Opera/LoRA、先验 Mel、技巧重绘等实验分支。

保留：可配置的 score/melody/neither 控制（random 为 score/melody 二选一）、masked flow L1 loss、AdamW（betas 0.9/0.95、eps 1e-6、weight decay 0.1）、梯度缩放/裁剪、线性 warmup + cosine、按 step 保存后再生成试听音频、torchrun FSDP 多卡。默认训练参数来自源项目。当前支持 FP32 或 FP16 autocast；不迁入源入口的 BF16 FSDP mixed-precision 策略，避免 Mel 缓冲区被整体转为低精度。

## 数据

编辑 `soulxsinger/config/train.yaml` 的 `data` 和 `train` 字段。索引兼容源格式：

```text
song_a:segment_0,segment_1,segment_2
song_b:segment_0,segment_1
```

对应文件是 `<pt_dir>/<song>/<segment>.pt`，每个文件是包含以下 CPU Tensor 的字典：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| waveform | `[L]` 或 `[1,L]` | 配置采样率下的单声道波形 |
| phoneme/note_pitch/note_type | `[N]` 或 `[1,N]` | token ID，必须兼容底模词表 |
| mel2note | `[T]` 或 `[1,T]` | token 序列位置 |
| f0 | `[T]` 或 `[1,T]` | Hz，与 mel2note 等长 |

数据必须预先完成音频和标注对齐。该入口不负责将原始音频转成 PT。多余字段不参与基础训练。超过 `max_sec` 或损坏的样本明确报错，请先清理索引；不沿用源代码的递归重新抽样。

按 seed 划分歌曲，前 `test_song_count` 首作为验证集，其余训练，歌曲不交叉。训练优先同歌不相邻片段，退化到不同片段，单片段歌曲才使用自身。验证优先跨歌并固定配对。设 `test_song_count: 0` 可关闭试听验证。默认 100 要求索引超过 100 首；小数据集需调小。

collate 分别补齐 prompt/target，返回单数 `prompt_mel_len` / `target_mel_len`，与现有 forward 一致（源 collate 的复数字段会导致长度 mask 未被读取）。

## 启动

在项目根目录、已安装项目依赖的 Python 环境运行：

```bash
python -m cli.train --config soulxsinger/config/train.yaml \
  --checkpoint pretrained_models/SoulX-Singer/model.pt

# 多卡，每卡的 batch_size 由 YAML 指定
 torchrun --standalone --nproc_per_node=2 -m cli.train \
  --config soulxsinger/config/train.yaml \
  --checkpoint pretrained_models/SoulX-Singer/model.pt
```

必须提供完整底模 checkpoint（含声码器），严格加载 `state_dict`、`weights` 或裸 state dict。训练不调用声码器；它仅用于试听，参数保持冻结。这里不会偷偷从头训练一个随机声码器。

单进程支持 CPU（设置 `train.fp16: false`），但完整模型通常需要 GPU。多进程要求 CUDA/NCCL，按 Transformer 层 FSDP FULL_SHARD。FP16 用 ShardedGradScaler；FSDP 梯度裁剪和完整优化器状态汇聚由 FSDP 接口完成。

## 输出和验证

`train.output_dir` 下产生：

- `metrics.jsonl`：step、epoch、训练 loss 和学习率，主进程写入。
- `checkpoints/step_N.pt`：完整 `state_dict`、optimizer、scheduler、scaler、配置和步数。
- `audio/step_N.wav`：固定验证配对的试听音频。验证 batch size 固定为 1，避免当前推理接口缺少 padding mask 的问题。

loss 从实际 flow 配置读取 sigma，并用 FP32 求和。与源实现一致，各 rank 的局部归一化 loss 取平均，不额外改成跨 rank 按帧数加权。

checkpoint 可直接交给现有 `cli/inference.py`。`--checkpoint` **只初始化模型权重**，不恢复训练进度、优化器或数据随机状态；暂未提供精确断点续训。试听验证不是全验证集 loss 或音质指标评测。模型规模若改动，推理时也要指定对应配置。

迁移只新增训练文件，保留已有推理流程。JSONL 替代源 TensorBoardX 日志，避免新增该依赖；训练代码不会自动下载或启动长训练。

## 检查

```bash
python -m unittest discover -s tests -v
python -m cli.train --help
```

测试覆盖 loss 的 sigma/mask、变长 collate、歌曲划分和固定验证配对，以及轻量模型两步训练、保存/加载、试听和训练状态恢复。完整模型多卡训练需在可用 GPU 与真实离线数据上验证。

## neither 接力训练实验

当前配置使用 `train.control: neither`、`infer.control: neither`，输出写入
`outputs/training_neither`。缺省这两个配置时，分别回退 `random` 和 `melody`。
训练支持 random/melody/score/neither，试听支持 melody/score/neither。

neither 在训练和推理中都忽略 prompt/target 的 F0 和 note_pitch，按 mel2note
和 note_type 的形状生成零输入；保留音高 embedding 的索引 0，embedding 输出
不一定为零。参考音频、音素、note_type 和 mel2note 仍参与条件编码，因此是
无显式音高条件生成，不是纯歌词生成。Dataset 仍要求原来的全部 PT 字段。

使用 `--checkpoint` 加载基模进行全参数接力训练（Mel 提取器和声码器继续冻结），
优化器和学习率重新初始化，不恢复之前的训练步数。模型参数键和形状不变。
独立推理使用 `python -m cli.inference ... --control neither`，默认仍为 melody；
neither 不支持 `--auto_shift` 或非零 `--pitch_shift`。SVC 模型不受影响。

训练开始前会先生成 `audio/step_0.wav`（不额外保存 step_0 checkpoint），
随后按原周期保存模型并生成试听。所有试听固定使用 `infer.seed`，未设置时
回退到 `train.seed`（再缺省为 24），与固定验证配对一起用于训练前后对比。
试听保留并恢复 Python、NumPy、CPU Torch 和当前 CUDA 设备的随机状态。
`test_song_count: 0` 时不执行训练前或训练期间的试听。
