# H20 物理一致性 / VideoPhy-2 / TRD 训练问题综合报告

更新时间：2026-04-18
合并来源：
- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/REPORT_h20_问题概述.md`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/REPORT_videophy2_h20_问题概述_legacy_full.md`

## 综合导读

这份综合报告把当前 H20/TRD 训练主线交接文档与旧版 VideoPhy-2 H20 全量排障报告合并到一个文件中。合并原则是：不删除原文、不压缩原文、不改写原始结论，只在最前面增加一段统一导读，说明两份报告之间的关系、阅读顺序与当前最重要的工程判断。

建议阅读顺序如下：

1. 先读“第一部分：当前 H20/TRD 训练主线交接”。这部分是最新主线，重点解释当前任务、三套代码关系、训练路径、已经修掉的问题、仍未解决的 `SIGFPE`、以及下一步定位建议。
2. 再读“第二部分：旧版 VideoPhy-2 H20 全量排障报告”。这部分保留了从 VideoPhy-2 AutoEval 全 0、全 1、坏输出、parser 修复、H20 路径、LingBot / Stage1 评测口径、Physics-IQ 后续方案，到最新 TRD-v1 `SIGFPE` 专项诊断的完整历史。

当前最重要的统一结论是：VideoPhy-2 AutoEval / generation-only / evaluation 路径已经不是当前主要阻塞；当前真正没有解决的是 H20 上 TRD-v1 student training path 在 backward 中触发 native `SIGFPE`。这条路径不同于 VideoREPA teacher/eval，也不同于 LingBot-base / LingBot-Stage1 generation-only 推理，因此“VideoREPA 和 LingBot Stage1 能跑”并不能证明当前 LoRA + patched Wan student backward 图一定能跑。

截至最新 probe，`SIGFPE` 已经排除了普通 OOM、TRD loss、loss NaN/Inf、DDP `find_unused_parameters`、多卡 NCCL/DDP、full checkpoint replay 作为主因；单卡 `world_size=1` 仍可复现。因此，当前最可疑区域是 Wan student backward 图中的 native CUDA path，尤其是 LoRA、memory-efficient Wan block patch、attention/flash-attn、bf16/dtype patch 与 H20 CUDA 后端的组合。

下面两部分均为原报告完整正文。为了保证可追溯性，原始标题、编号、代码块和结论均保留。

---

# 第一部分：当前 H20/TRD 训练主线交接（原 `REPORT_h20_问题概述.md` 全文）

# H20 物理一致性训练问题概述与完整交接

更新时间：截至 H20 训练日志 `2026-04-18 05:13`  
当前代码主线：`main`，最新已推送提交 `5cb9982 Use inner FFN checkpointing for Wan blocks`  
当前结论：VideoPhy-2 评测链路已不是主要阻塞点；当前主要阻塞是 TRD/Stage1 风格训练在 8 张 H20 上进入 student forward 后仍触发 native `SIGFPE`。

精确度说明：

```text
这份 REPORT_h20_问题概述.md 是“当前 H20/TRD 训练问题主线 + 必要历史”的交接文档。
旧版 VideoPhy-2 详细调试报告原文没有丢失，已恢复为：

/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/REPORT_videophy2_h20_问题概述_legacy_full.md

如果另一个聊天框需要完全复盘 VideoPhy-2 从全 0、全 1、空分数到最终跑通的所有细节，应同时读取本文件和 legacy_full 文件。
本文件中 VideoPhy-2 部分是压缩摘要；TRD/H20 训练 OOM/SIGFPE 部分是当前主线的完整交接。
```

## 0. 这份文档给谁看

这份文档的目标是让另一个 AI 聊天框或新的工程同学一次读完就能知道：

- 我们现在到底在做什么任务。
- 代码在哪里，训练实际在哪台机器跑。
- 原始 LingBot / 合作者 Stage1 / 当前 Physical Consistency 代码分别是什么关系。
- 从早期 LoRA 没有梯度，到双分支 OOM，再到 gradient checkpointing 和 Wan/ZeRO3/SDPA 组合 native crash，我们已经做过哪些尝试。
- 哪些问题已经修复，哪些问题仍然没有解决。
- 现在应该从哪里继续，而不是重复已经验证失败的路线。

## 1. 服务器、代码仓库与同步关系

### 1.1 Codex 所在机器

本报告由 Codex 在 hal 侧工作区编辑：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency
```

这个目录是当前开发、写报告、读日志、提交代码的主要工作区。

### 1.2 H20 训练机器

训练实际运行在 H20 服务器：

```text
host: instance-afs92r3e
training cwd: /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
conda env: /home/nvme03/workspace/world_model_phys/.conda_envs/phys-main
GPU: 8 x NVIDIA H20, 每张日志显示 total capacity 约 95.09 GiB
```

训练命令中通常使用：

```bash
GPU_LIST=0,1,2,3,4,5,6,7
REQUIRE_TRAIN_FLASH_ATTN=1
--num_gpus 8
--ulysses_size 8
```

### 1.3 GitHub 同步方式

hal 侧 Codex 修改代码后推送到 GitHub：

```text
origin git@github.com:jh5117-debug/world_model_phys.git
```

H20 侧通过：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
git pull origin main
git rev-parse --short HEAD
```

拉取最新代码再运行训练。

### 1.4 当前重要本地文件

hal 侧报告与日志：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/REPORT_h20_问题概述.md
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/REPORT_videophy2_h20_问题概述_legacy_full.md
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/train_trd_v1_low.log
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/train_trd_v1_dual.log
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/PROTOCOL_lingbot_videophy2_standard_eval.md
```

H20 侧训练脚本实际写入日志：

```text
/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_low.log
/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_dual.log
```

注意：当 `--model_type low` 时，更新的是 `train_trd_v1_low.log`，不是 `train_trd_v1_dual.log`。之前有一次误看 dual log，以为日志没更新，实际是因为切到了 single low run。

## 2. 当前任务到底是什么

我们不是单纯在跑 VideoPhy-2 评测，而是在做一个物理一致性训练链路：

```text
LingBot/Wan denoiser student
    + V-JEPA2 teacher features
    + TRD / relation distillation loss
    + CS:GO multi-view physical consistency dataset
    + LoRA tuning
    + 8 x H20 distributed DeepSpeed/Accelerate training
```

当前训练入口：

```text
src/physical_consistency/cli/train_trd_v1.py
src/physical_consistency/trainers/trd_v1.py
src/physical_consistency/trainers/stage1_components.py
scripts/run_train_trd_v1.sh
```

主要配置：

```text
configs/train_trd_v1_h20_perf.yaml
configs/accelerate_trd_v1_ultralowmem.yaml
configs/deepspeed_trd_v1_zero3_ultralowmem.json
```

当前典型运行命令：

```bash
GPU_LIST=0,1,2,3,4,5,6,7 \
REQUIRE_TRAIN_FLASH_ATTN=1 \
bash scripts/run_train_trd_v1.sh \
  --config configs/train_trd_v1_h20_perf.yaml \
  --accelerate_config configs/accelerate_trd_v1_ultralowmem.yaml \
  --model_type low \
  --project_name intro-example \
  --wandb_entity WorldModel_11 \
  --num_gpus 8 \
  --ulysses_size 8 \
  --num_frames 49 \
  --height 320 \
  --width 576 \
  --gradient_accumulation_steps 2 \
  --teacher_offload_after_encode true \
  --student_ffn_chunk_size 128 \
  --student_norm_chunk_size 128 \
  --student_lora_chunk_size 64 \
  --gradient_checkpointing true \
  --student_checkpoint_use_reentrant true \
  --max_train_micro_steps 1
```

`--max_train_micro_steps 1` 的目的只是 smoke/memory probe：只跑 1 个 micro step，确认能否完成第一次 forward/backward，不是正式训练。

## 3. 三套代码的关系

### 3.1 原 LingBot 代码

原 LingBot 代码路径：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/code/lingbot-world
```

关键文件：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/code/lingbot-world/wan/image2video.py
/home/hj/Multi-View-Physically-Consistent-World-Model/code/lingbot-world/wan/modules/model.py
```

原 LingBot 的 `image2video.py` 是推理逻辑，不是训练逻辑。它会加载两套完整去噪网络：

```text
low_noise_model
high_noise_model
```

推理时两套模型都会被配置为：

```python
model.eval().requires_grad_(False)
```

然后根据 timestep 选择当前需要的分支：

```text
t >= boundary -> high_noise_model
t < boundary  -> low_noise_model
```

如果 `offload_model=True`，推理时会把当前不需要的分支 offload 到 CPU，当前需要的分支放到 GPU。

结论：

- 原 LingBot 推理不是“训练一个、冻结另一个”。
- 它是两套模型都加载，但都 `eval().requires_grad_(False)`。
- active/inactive 分支通过 timestep 和 offload 机制切换。

### 3.2 合作者 Stage1 训练代码

合作者 Stage1 代码路径：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune
```

关键文件：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/train_lingbot_csgo.py
/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/run_train_dual.sh
/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/deepspeed_zero2.json
```

`train_lingbot_csgo.py` 文件开头明确写着：

```text
Trains ONE model (low_noise_model or high_noise_model) per invocation.
For dual-model MoE training, run this script twice via run_train_dual.sh.
```

也就是说合作者 Stage1 的低显存策略不是“同时加载 low/high，然后冻结另一个”，而是：

```text
训练 low 时，只加载 low_noise_model
训练 high 时，只加载 high_noise_model
需要 dual checkpoint 时，分别训练两次，最后保存到同一 output tree 的两个 subfolder
```

它的 timestep 分段逻辑：

```text
low_noise_model 处理 t < 947
high_noise_model 处理 t >= 947
```

这和我们后来做 single-branch TRD training 的方向是一致的：不要把两套完整 Wan denoiser 同时放进一个训练 runtime。

### 3.3 当前 Physical Consistency / TRD 代码

当前主工程路径：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency
```

H20 实际训练路径：

```text
/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
```

当前 TRD 训练最初支持 `model_type=dual`，后来加入了：

```text
--model_type low
--model_type high
```

目的是和合作者 Stage1 类似，可以只加载单个分支，降低显存。

## 4. VideoREPA 论文与我们的设置差异

用户提供的 VideoREPA 原文截图说明：

```text
VideoREPA 使用 CogVideoX 作为 base model。
生成视频为 49 frames，480 x 720。
VideoREPA-2B 在 32k OpenVid videos 上 finetune 4000 steps。
VideoREPA-5B 使用 LoRA，在 64k OpenVid videos 上 finetune 2000 steps。
默认 alignment depth 是 18。
实验使用 8 x NVIDIA A100 80GB，总 batch size 32。
```

容易误解的一点：

- VideoREPA 的论文说明 8 x A100 80GB 可训练，并不直接证明我们现在的 Wan/LingBot/TRD 设置也应该无痛跑通。
- 我们的 base model 不是同一套 CogVideoX，当前是 LingBot/Wan denoiser。
- 我们的 teacher 是 V-JEPA2，而 VideoREPA 截图中写的是 VideoMAEv2 alignment target encoder。
- 我们当前还叠加了 low/high MoE-style denoiser、Wan attention、Ulysses、DeepSpeed ZeRO3 CPU param offload、LoRA chunk、TRD feature hooks。

因此，“VideoREPA 8 x A100 80GB 可以跑”是重要参考，但不能直接等价为“当前 Wan/LingBot/TRD 在 H20 上一定只差一点配置”。

## 5. VideoPhy-2 评测链路状态

这份报告最早叫 `REPORT_videophy2_h20_问题概述.md`，因为最早的主线是 VideoPhy-2/AutoEval 在 H20 上跑不稳定。

现在的状态：

- VideoPhy-2 是 benchmark/protocol。
- VideoPhy-2-AutoEval 是自动 judge。
- 早期出现过全 `0`、全 `1`、空分数/坏输出等问题。
- 截至 `2026-04-12` 左右，VideoPhy-2-AutoEval 在 H20 上已经可以稳定跑完。
- `CS:GO test` 数据集直测成功。
- `LingBot-base` 和 `LingBot-Stage1` 的 `test_inf_result` 80 视频评测也已经成功。

相关路径：

```text
third_party/videophy/VIDEOPHY2
third_party/VideoREPA/evaluation/VIDEOPHY2
src/physical_consistency/eval/videophy2.py
scripts/run_videophy2_dataset_autoeval_parallel.sh
scripts/run_videophy2_lingbot_parallel.sh
```

当前真正阻塞训练推进的是 TRD 训练显存/native crash，不是 VideoPhy-2 AutoEval 本身。

## 6. 已经修复过的问题

### 6.1 LoRA 没有梯度回传

早期问题：

```text
LoRA 参数被注入到 Wan linear layers 后，训练过程中发现 LoRA 没有正常梯度回传。
```

核心原因之一：

```text
PyTorch reentrant gradient checkpointing 要求至少一个输入 requires_grad=True。
LoRA tuning 下 base/student 输入可能本身是 frozen tensor。
如果整块 checkpoint 的输入都不 requires_grad，reentrant checkpoint 可能让参数梯度丢失。
```

已做修复：

- `apply_gradient_checkpointing()` 在 reentrant 模式下，如果 checkpoint args 中没有任何 tensor requires_grad，会给 hidden state 加一个 local grad anchor。
- 增加测试 `test_reentrant_gradient_checkpointing_preserves_parameter_grads_without_input_grads`。
- LoRA wrapper 保证 `lora_A/lora_B` 是 trainable，base linear 冻结。
- 增加 LoRA state dict / merge / chunked forward 相关测试。

相关文件：

```text
src/physical_consistency/trainers/stage1_components.py
tests/test_train_args.py
tests/test_stage1_components.py
```

当前结论：

```text
LoRA 无梯度问题已经修复，不是当前 H20 训练崩溃的主要原因。
```

### 6.2 Wan RoPE dtype 导致训练内存暴涨

问题：

```text
Wan rope_apply 会生成较大的 fp32 q/k 临时张量，长序列下显存压力明显。
```

已做修复：

```text
Patched Wan rope_apply to preserve q/k dtype for training memory
```

相关提交：

```text
7697661 Preserve dtype in Wan RoPE during training
```

### 6.3 Wan modulation / norm / FFN 显存优化

已做修复：

- memory-efficient modulation patch。
- Wan sequence norm chunk patch。
- FFN chunking。
- LoRA adapter output chunking。

日志中可见：

```text
Memory-efficient modulation patched 40 blocks for low_noise_model
Memory-efficient Wan sequence norms patched 281 modules for low_noise_model
Applied standard LoRA to 560 linear layers
```

相关提交：

```text
ff6d84d Chunk Wan sequence norms during TRD training
dbe69ca Chunk LoRA adapter outputs during TRD training
```

### 6.4 lowmem / ultralowmem DeepSpeed 配置

已新增配置：

```text
configs/accelerate_trd_v1_lowmem.yaml
configs/deepspeed_trd_v1_zero3_lowmem.json
configs/accelerate_trd_v1_ultralowmem.yaml
configs/deepspeed_trd_v1_zero3_ultralowmem.json
```

ultralowmem 当前关键设置：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "reduce_bucket_size": 10000000,
    "allgather_bucket_size": 1000000,
    "stage3_prefetch_bucket_size": 1000000,
    "stage3_param_persistence_threshold": 0,
    "stage3_max_live_parameters": 1000000,
    "stage3_max_reuse_distance": 1000000
  },
  "bf16": {
    "enabled": true
  }
}
```

相关提交：

```text
30a9b9a Add low-memory TRD ZeRO3 config
b501c8e Add ultra-low-memory TRD DeepSpeed config
```

### 6.5 训练 micro-step smoke/probe

新增：

```text
--max_train_micro_steps 1
```

用于只跑到第一个 micro step，避免每次都完整进入 checkpoint/validation。

相关提交：

```text
1c085a6 Add TRD memory probe micro-step limit
```

注意历史命令里出现过误写：

```text
--max_train_micro_steps 1false
--max_train_micro_steps 1reentrant false
```

这种 shell 拼写会污染实际参数。后续必须明确写成：

```bash
--max_train_micro_steps 1
```

### 6.6 single-branch TRD training

早期 `model_type=dual` 会同时构建：

```text
low_noise_model
high_noise_model
```

两套都是完整 Wan denoiser。即使只训练 LoRA，每套 base 权重仍然巨大，ZeRO3 还要做参数 gather/offload。

已新增：

```text
--model_type low
--model_type high
```

single-branch 模式只加载一个 denoiser 分支，和合作者 Stage1 的训练方式一致。

相关提交：

```text
54ed63b Support single-branch TRD training
```

效果：

```text
dual after_accelerator_prepare reserved ≈ 69.97 GiB
low  after_accelerator_prepare reserved ≈ 35.22 GiB
```

这说明 single-branch 确实把 ZeRO3 初始化阶段的 reserved memory 几乎砍半。

## 7. 历史失败与对应结论

### 7.1 dual 模式 OOM

命令特征：

```text
--model_type dual
--num_gpus 8
--num_frames 49
--height 320
--width 576
--student_ffn_chunk_size 256 或 512
```

完整日志：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/train_trd_v1_dual.log
```

关键日志：

```text
[TRAIN PLAN] epochs=5 micro_steps_per_epoch=209 optimizer_steps_per_epoch=105 total_optimizer_steps=525 grad_accum=2 dataset_samples=1670 world_size=8
Memory-efficient modulation patched 40 blocks for low_noise_model
Memory-efficient modulation patched 40 blocks for high_noise_model
[GPU MEM] after_accelerator_prepare allocated=0.64 GiB reserved=69.97 GiB
[SEQ GEOM] num_frames=49 latent_grid=(13,40,72) patch_size=(1,2,2) seq_len=9360
[GPU MEM] after_teacher_encode allocated=0.87 GiB reserved=1.07 GiB max_allocated=11.59 GiB
torch.OutOfMemoryError: Tried to allocate 136.00 MiB
```

OOM 栈位置：

```text
DeepSpeed ZeRO3 parameter all_gather / parameter_offload
stage1_components.py -> _run_ffn_residual -> LoRALinear.base -> DeepSpeed pre_forward hook
```

结论：

```text
dual 模式下两套完整 denoiser 带来的参数 gather/offload 峰值过高。即使 H20 单卡约 95GiB，第一步 student forward 仍然在 FFN/base linear gather 时爆掉。
```

### 7.2 single low 无 checkpoint 仍 OOM

命令特征：

```text
--model_type low
--gradient_checkpointing false 或未稳定生效
--num_frames 49
--height 320
--width 576
```

关键日志：

```text
[GPU MEM] after_accelerator_prepare reserved=35.22 GiB
[GPU MEM] after_teacher_encode max_allocated=11.52 GiB
torch.OutOfMemoryError: Tried to allocate 136.00 MiB
```

结论：

```text
single low 已经降低 ZeRO3 初始化 reserved memory，但第一次 student forward 的 activation + ZeRO3 gather 峰值仍然接近 95GiB，仍会 OOM。
```

### 7.3 full-block gradient checkpointing 触发 native SIGFPE

尝试：

```text
--gradient_checkpointing true
--student_checkpoint_use_reentrant false
```

现象：

```text
after_teacher_encode 后，没有 Python OOM stack，直接 ChildFailedError
Root Cause: Signal 8 (SIGFPE)
```

后续尝试：

```text
--student_checkpoint_use_reentrant true
```

仍然：

```text
Signal 8 (SIGFPE)
```

早期推断：

```text
不是单纯 reentrant/non-reentrant 差异。
full-block checkpoint 把 Wan block 里的 attention/SDPA/flash 路径也包进去了，在 H20 + ZeRO3 + Wan attention 下容易触发 native kernel crash。
```

### 7.4 non-reentrant metadata / early-stop 相关修复

已做修复：

```text
e571e8d Stabilize Wan checkpointing with ZeRO3
```

内容：

- non-reentrant checkpoint 加 `determinism_check="none"`，避免 ZeRO3 sharded empty tensor metadata mismatch。
- non-reentrant checkpoint 使用 `set_checkpoint_early_stop(False)`，避免重算提前结束导致 DeepSpeed gather/release hook 不平衡。
- 修复 memory-efficient Wan block patch 中 camera/control injection 逻辑。

验证：

```bash
PYTHONPATH=src pytest -q tests/test_train_args.py tests/test_stage1_components.py
```

当时结果：

```text
48 passed
```

但 H20 上仍然在 student forward 附近 native `SIGFPE`。

结论：

```text
metadata/early-stop 是必须修的潜在问题，但不是当前唯一根因。
```

### 7.5 inner FFN checkpointing 修复

最新修复：

```text
5cb9982 Use inner FFN checkpointing for Wan blocks
```

修改点：

```text
apply_gradient_checkpointing() 遇到已经 memory-efficient patched 的 Wan block 时，不再整体包 block.forward。
改为在 memory-efficient Wan forward 内部只 checkpoint FFN residual/chunk。
attention/self_attn/cross_attn 不进入 checkpoint。
```

日志确认本次 H20 low run 已经使用了这条新路径：

```text
[2026-04-18 05:12:12,771] INFO physical_consistency.trainers.stage1_components:
Gradient checkpointing enabled inner FFN checkpointing for 40 memory-efficient Wan blocks in low_noise_model
(use_reentrant=True; attention remains outside checkpoint)
```

本地测试：

```bash
PYTHONPATH=src pytest -q tests/test_stage1_components.py tests/test_train_args.py
```

结果：

```text
49 passed
```

当前结论：

```text
full-block checkpoint 这个问题已经规避，但 H20 上仍然 SIGFPE。
因此当前 bug 已经不再能简单归因于“attention 被 checkpoint 包住”。
更可能是 H20 上 Wan attention/SDPA/flash 路径本身与 ZeRO3/offload/长序列/并行组合仍存在 native kernel 级崩溃。
```

## 8. 当前最新失败状态

最新命令：

```bash
GPU_LIST=0,1,2,3,4,5,6,7 REQUIRE_TRAIN_FLASH_ATTN=1 bash scripts/run_train_trd_v1.sh \
  --config configs/train_trd_v1_h20_perf.yaml \
  --accelerate_config configs/accelerate_trd_v1_ultralowmem.yaml \
  --model_type low \
  --project_name intro-example \
  --wandb_entity WorldModel_11 \
  --num_gpus 8 \
  --ulysses_size 8 \
  --num_frames 49 \
  --height 320 \
  --width 576 \
  --gradient_accumulation_steps 2 \
  --teacher_offload_after_encode true \
  --student_ffn_chunk_size 128 \
  --student_norm_chunk_size 128 \
  --student_lora_chunk_size 64 \
  --gradient_checkpointing true \
  --student_checkpoint_use_reentrant true \
  --max_train_micro_steps 1
```

完整日志：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/train_trd_v1_low.log
```

关键日志：

```text
[2026-04-18 05:10:45,304] [TRAIN PLAN] epochs=5 micro_steps_per_epoch=209 optimizer_steps_per_epoch=105 total_optimizer_steps=525 grad_accum=2 dataset_samples=1670 world_size=8
[2026-04-18 05:12:11,780] Memory-efficient modulation patched 40 blocks for low_noise_model (ffn_chunk_size=128)
[2026-04-18 05:12:12,771] Gradient checkpointing enabled inner FFN checkpointing for 40 memory-efficient Wan blocks in low_noise_model (use_reentrant=True; attention remains outside checkpoint)
[2026-04-18 05:12:41,307] [GPU MEM] after_accelerator_prepare device=cuda:0 allocated=0.57 GiB reserved=35.22 GiB max_allocated=0.57 GiB
[2026-04-18 05:12:43,854] [GPU MEM] after_teacher_load device=cuda:0 allocated=0.57 GiB reserved=0.58 GiB max_allocated=0.57 GiB
[2026-04-18 05:12:59,419] [SEQ GEOM] num_frames=49 latent_grid=(13,40,72) patch_size=(1,2,2) seq_len=9360
[2026-04-18 05:13:00,151] [GPU MEM] after_teacher_encode device=cuda:0 allocated=0.80 GiB reserved=0.95 GiB max_allocated=11.52 GiB
Root Cause: Signal 8 (SIGFPE) received by PID 2636835
```

还有连续 warnings：

```text
FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated
```

这说明崩溃前确实走到了 Wan/attention/SDPA 相关路径。

当前仍存在的 bug：

```text
single low + 49 frames + 320x576 + ZeRO3 ultralowmem + inner FFN checkpointing 仍然在第一次 student forward 附近 native SIGFPE。
没有 Python traceback，torch elastic 只报告 Signal 8。
```

## 9. 当前不要重复的路线

### 9.1 不要再把“降低序列长度”作为第一反应

用户明确要求：

```text
不要每次去降一点序列长度。
应该找 gradient checkpointing / ZeRO3 / Wan / flash / SDPA 为什么组合不稳定，修 bug 或定位根因。
```

短序列可以作为定位手段，但不能作为最终路线。如果只是把 `num_frames`、`height`、`width` 不断降低，可能绕过问题，却无法解释为什么 VideoREPA/Stage1 类似工作能训练，也无法得到目标设置下的可用方案。

### 9.2 不要把 dual 分支误认为只是“冻结另一个分支”

`low_noise_model` 和 `high_noise_model` 是两套完整 denoising network。dual 训练同时加载两套，显存/ZeRO3 参数 gather 压力接近翻倍。

合作者 Stage1 的做法不是冻结另一个分支，而是单次只加载一个分支训练。

### 9.3 不要误判最新 `5cb9982` 没有生效

用户 tail 中只显示：

```text
Applying block-level gradient checkpointing for low_noise_model
```

但完整 low log 中已经有：

```text
Gradient checkpointing enabled inner FFN checkpointing for 40 memory-efficient Wan blocks...
```

所以最新代码确实生效了。当前 `SIGFPE` 是 inner FFN checkpointing 后仍存在的新状态。

## 10. 当前最可能的根因方向

当前证据链：

- 无 checkpoint：OOM，Python 栈显示在 ZeRO3 all_gather / LoRALinear.base / FFN。
- full-block checkpoint：从 OOM 变为 native SIGFPE。
- full-block checkpoint 的 reentrant 和 non-reentrant 都 SIGFPE。
- 修复 non-reentrant metadata/early-stop 后仍 SIGFPE。
- 改为 inner FFN checkpoint，attention 不进 checkpoint 后仍 SIGFPE。
- 崩溃点仍在 `after_teacher_encode` 后，进入 student forward 后不久。
- 崩溃前出现多次 `torch.backends.cuda.sdp_kernel()` warning。

因此当前最可能方向：

```text
Wan attention/SDPA/flash-attn 路径在 H20 + 8-way distributed + ZeRO3 CPU param offload + 长序列 seq_len=9360 的组合下存在 native kernel crash。
```

仍需确认的具体点：

- 是 flash-attn kernel 本身的问题，还是 PyTorch SDPA fallback/dispatch 的问题。
- 是 Ulysses/sequence parallel 与 attention kernel 的组合问题，还是普通 distributed 也会触发。
- 是 ZeRO3 参数 offload hook 与 attention 前后某个 module gather/release 冲突，还是 attention 输入 shape/dtype/mask 在 H20 上触发非法路径。
- 是 `REQUIRE_TRAIN_FLASH_ATTN=1` 强制 flash-attn 后才触发，还是不强制也触发。
- 是 H20 驱动/CUDA/PyTorch/flash-attn 版本组合问题。

## 11. 后续建议的定位方式

这些是定位建议，不是要求立刻改代码。

### 11.1 首先抓 native 栈

建议在 H20 上尽量开启：

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO
```

如果可行，也可以允许 core dump：

```bash
ulimit -c unlimited
```

目标是让 `SIGFPE` 不再只是 elastic 的 `Signal 8`，而是能看到 C++/CUDA 层到底在哪个 kernel 或哪个 attention path 崩。

### 11.2 区分 flash-attn 与 PyTorch SDPA

当前命令用了：

```bash
REQUIRE_TRAIN_FLASH_ATTN=1
```

后续应设计 diagnostic run 来区分：

```text
强制 flash-attn
禁用 flash-attn / 使用 math 或 mem_efficient SDPA
```

这不是为了退回慢路径训练，而是为了确认 native crash 的归属。

### 11.3 区分 attention crash 与 FFN/ZeRO3 crash

现在 inner FFN checkpoint 已经避免了 full-block checkpoint。下一步如果还要改代码，应该考虑加极细粒度日志或 NVTX around：

```text
before self_attn
after self_attn
before cross_attn
after cross_attn
before FFN checkpoint chunk
after FFN checkpoint chunk
```

当前日志只能知道在 `after_teacher_encode` 后、`max_train_micro_steps` 退出前崩溃，还不能精确到 self-attn/cross-attn/FFN。

### 11.4 继续对比合作者 Stage1

合作者 Stage1 可以重点对比：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/train_lingbot_csgo.py
/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/deepspeed_zero2.json
```

值得注意的差异：

- 合作者 Stage1 说是 `zero2` 文件名，但 JSON 里实际是 `"stage": 3`。
- 它使用 CPU param offload 和 CPU optimizer offload。
- 它 `overlap_comm=true`，bucket size 很大：`5e8`。
- 它 full-block checkpoint 用 `use_reentrant=False`。
- 它训练一套分支，不是 dual。
- 它没有当前 TRD 的 V-JEPA2 teacher feature relation loss 和 student target block hook。

当前我们的 ultralowmem：

- CPU param offload。
- 没有 optimizer offload。
- `overlap_comm=false`。
- bucket 极小：`allgather_bucket_size=1e6`、`stage3_prefetch_bucket_size=1e6`。
- inner FFN checkpoint。
- 有 V-JEPA2 teacher、TRD loss、feature hook。

## 12. 重要提交时间线

当前 `git log --oneline -n 12`：

```text
5cb9982 Use inner FFN checkpointing for Wan blocks
e571e8d Stabilize Wan checkpointing with ZeRO3
54ed63b Support single-branch TRD training
b501c8e Add ultra-low-memory TRD DeepSpeed config
1c085a6 Add TRD memory probe micro-step limit
dbe69ca Chunk LoRA adapter outputs during TRD training
30a9b9a Add low-memory TRD ZeRO3 config
dc7108f Allow TRD resolution CLI overrides
ff6d84d Chunk Wan sequence norms during TRD training
7f291dd Allow training LoRA from later blocks
7697661 Preserve dtype in Wan RoPE during training
7acf981 Disable checkpoint metadata check for ZeRO3
```

这些提交大致对应：

```text
7acf981: 处理 ZeRO3 + checkpoint metadata mismatch。
7697661: Wan RoPE 保持 dtype，避免 q/k fp32 暴涨。
7f291dd: 允许从更后面的 block 开始 LoRA 训练。
ff6d84d: chunk Wan norm。
dc7108f: CLI 支持分辨率覆盖。
30a9b9a: lowmem ZeRO3 config。
dbe69ca: LoRA adapter output chunk。
1c085a6: max_train_micro_steps smoke/probe。
b501c8e: ultralowmem ZeRO3 config。
54ed63b: single-branch low/high training。
e571e8d: ZeRO3 checkpointing 稳定性修复、camera injection 修复。
5cb9982: memory-efficient Wan block 改用 inner FFN checkpoint，attention 不进 checkpoint。
```

## 13. 当前状态一句话总结

截至当前：

```text
VideoPhy-2 AutoEval 评测链路已经基本跑通。
LoRA 无梯度、RoPE dtype、norm/FFN/LoRA chunk、single-branch、ultralowmem ZeRO3、checkpoint metadata/early-stop、full-block checkpoint 这些问题都已经逐步处理。
当前仍未解决的是：H20 上 single low 分支，在 49 frames / 320x576 / seq_len=9360 / ZeRO3 ultralowmem / inner FFN checkpointing / flash-attn required 的第一次 student forward 附近触发 native SIGFPE。
```

后续接手的人不要从“是不是 dual 加载两套模型”或“是不是 LoRA 没梯度”重新开始排查；这些已经有明确结论。应优先围绕 `Wan attention/SDPA/flash-attn + H20 + ZeRO3/offload + 长序列` 的 native crash 做更精确定位。

---

# 第二部分：旧版 VideoPhy-2 H20 全量排障报告（原 `REPORT_videophy2_h20_问题概述_legacy_full.md` 全文）

# VideoPhy-2 在 H20 上的问题概述与当前结论

## 1. 文档目的

这份文档用于完整记录截至目前为止，我们在 H20 机器上接入、调试和评估 `VideoPhy-2 / VideoPhy-2-AutoEval` 的全过程，包括：

- VideoPhy-2 本身是什么
- 我们当前项目里是怎么接入它的
- H20 上从 0 开始的环境与代码调试过程
- 为什么最开始会出现全 `0`
- 为什么后面又会变成全 `1`
- 为什么当前 `VideoPhy-2-AutoEval` 结论仍然不可用
- 如果 LingBot / LingBot-Stage1 要做标准 VideoPhy-2 benchmark，正确 protocol 应该是什么

这份文档的目标不是展示“某次偶然跑通的数字”，而是把目前已经确认的技术事实和实验边界说清楚。

### 1.1 2026-04-12 最新状态说明

这份文档保留了从“全 `0` -> 全 `1` -> 空分数/坏输出”的完整调试历史。

但截至 `2026-04-12` 晚上的最新状态已经是：

- `VideoPhy-2-AutoEval` 在 H20 上已经可以稳定跑完
- `CS:GO test` 数据集直测已经全量成功
- `LingBot-base` 与 `LingBot-Stage1` 的 `test_inf_result` 80 视频评测也已经全量成功
- 最开始“几乎全是 `1`”的问题已经定位并修复

因此，后文中凡是写到“当前不可用”或“当前结论仍不可信”的段落，都应理解为：

- **那是中间调试阶段的历史结论**
- **不是 2026-04-12 修复完成之后的最终状态**

## 2. VideoPhy-2 是什么

### 2.1 基本定义

`VideoPhy-2`（也常写作 `VideoPhy2` 或 `VideoPhy-2 benchmark`）本质上是一个 **视频生成物理一致性评测基准**，不是视频生成模型。

它主要关注两类分数：

- `Semantic Adherence (SA)`：
  - 生成视频是否符合给定文本描述
- `Physical Commonsense (PC)`：
  - 视频中的运动和交互是否符合直觉物理和常识物理

官方还定义了：

- `joint score = fraction(SA >= 4 and PC >= 4)`

从官方 README 可以看到：

- `SA` 输出 `1-5`
- `PC` 输出 `1-5`
- `Rule` 任务输出 `0/1/2`

见：

- [VIDEOPHY2/README.md](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/videophy/VIDEOPHY2/README.md#L52)

### 2.2 AutoEval 是什么

`VideoPhy-2-AutoEval` 是 benchmark 附带的一个 **自动打分器 / judge**。

它的输入是：

- `SA`：
  - `videopath`
  - `caption`
- `PC`：
  - `videopath`

它的作用是：

- 看视频
- 根据 prompt 或物理合理性给出打分

所以要区分：

- `VideoPhy-2`：
  - benchmark / protocol
- `VideoPhy-2-AutoEval`：
  - benchmark 的自动评审器

只用 AutoEval 去打自己的一批视频，并不自动等于“做了标准 VideoPhy-2 benchmark”。

## 3. 当前项目里是如何接入 VideoPhy-2 的

### 3.1 当前 wrapper 的逻辑

当前项目里，VideoPhy-2 的 Python wrapper 主要在：

- [videophy2.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/eval/videophy2.py)

其中关键逻辑是：

- 构造 VideoPhy-2 所需输入 CSV
- `SA` 时从 manifest 的 `prompt` 列取文本
- `PC` 时只传视频路径
- 调用官方 `third_party/videophy/VIDEOPHY2/inference.py`
- 汇总 `SA mean` / `PC mean` / `joint`

关键位置：

- [build_videophy2_input_csv](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/eval/videophy2.py#L51)
- [summarize_videophy2_outputs](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/eval/videophy2.py#L145)

### 3.2 当前两个主要入口

项目里目前有两类入口：

- 数据集直接 AutoEval：
  - [run_videophy2_dataset_autoeval_parallel.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_videophy2_dataset_autoeval_parallel.sh)
- LingBot / Stage1 生成视频后再评测：
  - [run_videophy2_lingbot_parallel.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_videophy2_lingbot_parallel.sh)

它们的差别是：

- `dataset_autoeval`：
  - 直接拿已有视频打分
- `lingbot_parallel`：
  - 假设模型已经生成了视频，再对生成结果打分

## 4. H20 接入与代码演化时间线

### 4.1 H20 路径与启动脚本接入

早期接入阶段的几个关键 commit：

- `89756a0` Add H20-2 path defaults and evaluation launchers
- `48dbbb6` Set default H20-2 LingBot code paths

它们主要解决：

- H20 上的路径默认值
- LingBot / Stage1 / VideoPhy wrapper 的启动入口

### 4.2 VideoPhy 运行期兼容修复

后续为了让 VideoPhy-2 在 H20 上至少能运行，又加入了几次修复：

- `7e6f83d` Stabilize VideoPhy CPU temporal conv on H20
- `e5e0727` Allow VideoPhy inference dtype override
- `4cf4498` Improve VideoPhy launcher visibility and GPU cleanup

这些修改主要处理了：

- H20 上的 `SIGFPE`
- `bfloat16` / `float16` 的不稳定问题
- 并行脚本缺少中间日志、GPU 清理不便的问题

### 4.3 H20 服务器上的关键路径与文件布局

根据当前 `path_config_cluster.env`、早期路径探测记录以及后续实际运行日志，H20 上当前这条链路的关键路径可以整理为：

- 项目根目录：
  - `/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys`
- 资产根目录：
  - `/home/nvme03/workspace/world_model_phys/PHYS`
- 权重目录：
  - `/home/nvme03/workspace/world_model_phys/PHYS/weight`
- 数据集目录：
  - `/home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3`
- VideoPhy-2 AutoEval 权重：
  - `/home/nvme03/workspace/world_model_phys/PHYS/weight/videophy_2_auto`
- LingBot-base 权重：
  - `/home/nvme03/workspace/world_model_phys/PHYS/weight/Lingbot-base`
- LingBot-Stage1 权重：
  - `/home/nvme03/workspace/world_model_phys/PHYS/weight/Lingbot-Stage1`
- conda 环境：
  - `/home/nvme03/workspace/world_model_phys/.conda_envs/phys-videophy`
- 代理脚本：
  - `/home/nvme01/clash-for-linux/clash.sh`

结合当前配置文件：

- [path_config_cluster.env](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/configs/path_config_cluster.env#L1)

可以得到 H20 上当前应该存在的核心树形结构：

```text
/home/nvme03/workspace/world_model_phys/PHYS
├── Dataset/
│   └── processed_csgo_v3/
│       ├── metadata_train.csv
│       ├── metadata_val.csv
│       ├── train/
│       ├── val/
│       │   └── clips/
│       │       └── <clip_dir>/video.mp4
│       └── test/
│           ├── bear/
│           ├── bike-packing/
│           └── blackswan/
├── weight/
│   ├── Lingbot-base/
│   │   ├── low_noise_model/
│   │   ├── high_noise_model/
│   │   ├── google/umt5-xxl/
│   │   ├── Wan2.1_VAE.pth
│   │   └── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Lingbot-Stage1/
│   │   ├── low_noise_model/
│   │   ├── high_noise_model/
│   │   ├── google/umt5-xxl/
│   │   ├── Wan2.1_VAE.pth
│   │   └── models_t5_umt5-xxl-enc-bf16.pth
│   └── videophy_2_auto/
│       ├── config.json
│       ├── generation_config.json
│       ├── tokenizer.model
│       ├── pytorch_model-00001-of-00002.bin
│       └── pytorch_model-00002-of-00002.bin
└── world_model_phys/
    ├── configs/
    ├── scripts/
    ├── src/
    ├── third_party/
    │   ├── videophy/VIDEOPHY2/
    │   └── VideoREPA/
    └── runs/
        └── eval/
```

补充说明：

- 上面这棵树是基于当前配置和实际运行记录整理出的 **应有结构**。
- 如果需要完全以服务器当前真实状态为准，应执行下一节的查找命令。

### 4.4 H20 上查看真实文件树的命令

如果 H20 上安装了 `tree`，建议执行：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS
tree -L 2
tree -L 2 weight
tree -L 3 Dataset/processed_csgo_v3
tree -L 3 world_model_phys
```

如果 H20 上没有 `tree`，可以用下面这些命令替代：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS

find weight -maxdepth 2 \\( -type d -o -type f \\) | sort
find Dataset/processed_csgo_v3 -maxdepth 3 \\( -type d -o -type f \\) | sort | sed -n '1,200p'
find world_model_phys -maxdepth 3 \\( -type d -o -type f \\) | sort | sed -n '1,200p'
```

如果只想确认关键资产是否都在，可以执行：

```bash
ls -lh /home/nvme03/workspace/world_model_phys/PHYS/weight/videophy_2_auto
ls -lh /home/nvme03/workspace/world_model_phys/PHYS/weight/Lingbot-base
ls -lh /home/nvme03/workspace/world_model_phys/PHYS/weight/Lingbot-Stage1
ls -lh /home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/metadata_val.csv
ls -lh /home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/val/clips | sed -n '1,20p'
```

如果需要查看某个具体 clip 是否存在视频文件，可以执行：

```bash
find /home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/val/clips -name 'video.mp4' | sed -n '1,20p'
```

### 4.5 早期 H20 路径探测时发现的问题

在最早的 H20 路径探测中，我们把服务器上的关键候选路径扫描成了一个记录文件：

- [h20_2_path_probe_20260409_163904.txt](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/h20_2_path_probe_20260409_163904.txt#L1)

这次探测确认了几件很重要的事：

- `Lingbot-base` 和 `Lingbot-Stage1` 的主要权重目录是存在的
- `processed_csgo_v3/val/clips/.../video.mp4` 数据集视频是存在的
- `third_party/videophy/VIDEOPHY2/inference.py` 是存在的
- 但当时：
  - `videophy_2_auto` 还不存在
  - `runs/eval` 还不存在
  - `LINGBOT_CODE_DIR` 和 `FINETUNE_CODE_DIR` 当时还是空的

这意味着在最开始阶段，服务器上的状态其实是：

- 数据集已有
- Base/Stage1 主要权重已有
- VideoPhy-2 代码已有
- 但 VideoPhy-2 AutoEval 权重缺失
- 评测输出目录缺失
- 原始 LingBot 代码和 finetune eval 代码路径也还没有完全配置好

这也是为什么最开始在 H20 上接通整条链路时，主要困难不是某一个单点 bug，而是：

- 资源路径不完整
- 权重不完整
- 代码路径不完整
- 环境也还没修好

## 5. H20 上从 0 开始的调试过程

### 5.1 前期数据与权重传输的困难

在早期接入阶段，最大的工程障碍之一不是代码逻辑本身，而是：

- H20 服务器上关键资产并不齐全
- 大文件传输成本高
- 单纯依赖手动拷贝或本地中转非常不稳定

从路径探测记录可以明确看出，当时：

- `processed_csgo_v3` 数据集已经在 H20 上
- `Lingbot-base` / `Lingbot-Stage1` 主要权重已经在 H20 上
- 但 `videophy_2_auto` 不在 H20 上

也就是：

- benchmark 所需的视频数据在
- LingBot 相关权重也在
- 但 VideoPhy-2 AutoEval 的关键权重缺失

这直接导致一个现实问题：

- 想继续推进就必须把 `videophy_2_auto` 这个 10GB+ 级别的模型资产补到 H20 上

而这一阶段最大的困难是：

- 从本地机器到 H20 传超大文件很麻烦
- 传输过程慢、容易中断
- 即使想先在本地下载再中转，也会增加很多额外时间成本

因此，后面的关键策略转变不是“继续硬传”，而是：

- **改成在 H20 上直接从 Hugging Face 拉取权重**

这一步是整个调试过程里非常关键的转折点。

### 5.2 最终成功的方法：通过 HF 镜像在 H20 上直接下载 `videophy_2_auto`

最终成功的方法不是本地中转，而是：

1. 在 H20 上启用代理
2. 把 `HF_ENDPOINT` 指到镜像
3. 直接在 H20 上运行 `snapshot_download`

工作命令序列如下：

```bash
conda activate /home/nvme03/workspace/world_model_phys/.conda_envs/phys-videophy

source /home/nvme01/clash-for-linux/clash.sh && proxy_on
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="videophysics/videophy_2_auto",
    local_dir="/home/nvme03/workspace/world_model_phys/PHYS/weight/videophy_2_auto",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
```

最终在 H20 上成功下载了：

- `/home/nvme03/workspace/world_model_phys/PHYS/weight/videophy_2_auto`

并确认存在两个大分片：

- `pytorch_model-00001-of-00002.bin`
- `pytorch_model-00002-of-00002.bin`

也就是说，到这一步我们才真正把：

- 数据
- LingBot 权重
- VideoPhy-2 AutoEval 权重

这三类关键资产同时凑齐。

### 5.3 环境缺包与版本兼容

最早遇到的是环境问题，不是模型本身问题。

具体包括：

- 缺 `filelock`
- 错装了 `huggingface_hub 1.x`，与 `transformers==4.28.1` 不兼容
- 缺 `sentencepiece`

后续修正为：

- `huggingface_hub==0.36.2`
- 安装 `regex`
- 安装 `safetensors`
- 安装 `sentencepiece`
- 确认 `decord`、`pandas`、`transformers`、`torch` 等依赖都可 import

这里的一个重要教训是：

- 只要是这种老版本 `transformers` + 官方第三方 repo 的组合，环境里不要随手把 `huggingface_hub` 升到最新大版本
- 对 VideoPhy-2 这条链路来说，先“环境能稳 import 并能 load model”比“追新版本”重要得多

### 5.4 初步验证：环境可用但推理失败

在完成依赖安装后，我们确认了：

- `decord` 能正常读取视频
- tokenizer / processor / model load 都正常

也就是说：

- 不是数据路径问题
- 不是缺包问题
- 不是基本模型加载问题

但并行 VideoPhy-2 评测依然失败，最初表现为：

- `SIGFPE`
- 或者子进程直接退出

这一步推动了对 dtype 和 conv hack 的修复。

### 5.5 并行运行期的额外困难与解决

除了模型与环境本身的问题，H20 上还遇到了若干“工程运行期”困难，这些后来也都被逐步解决了。

#### 5.5.1 缺少中间日志，看起来像“没反应”

在并行脚本的早期版本里：

- 启动后几乎没有中间输出
- 用户很难判断是卡死、在运行，还是已经失败

因此后来在 launcher 中加入了：

- 配置打印
- shard 启动信息
- 日志路径打印
- worker PID 打印
- 周期性 `[WAIT]` 心跳输出

对应修复：

- `4cf4498` Improve VideoPhy launcher visibility and GPU cleanup

#### 5.5.2 GPU 4-7 经常被占用

在 H20 上实际运行时，4-7 号卡常常已经有其他 `python` 进程占用。

为了解决这个问题，我们采用了两层办法：

- 手动：
  - 通过 `nvidia-smi` 查看 PID
  - `kill -9 <pid>` 清掉目标 GPU 上的进程
- 自动：
  - 在 launcher 中加入目标 GPU 进程扫描与可选自动清理

同样属于 `4cf4498` 这一批运维改造。

#### 5.5.3 `tree`、`ffmpeg` 等系统工具并不一定可用

在 H20 上并不是所有常见系统工具都默认存在。

例如：

- `ffmpeg` 不存在

这导致我们在测试 `bear / bike-packing / blackswan` 三个自然视频样本时，最初无法直接把图片序列打成 `mp4`。

最后的解决办法是：

- 不依赖 `ffmpeg`
- 直接用当前 conda 环境里已有的 `cv2.VideoWriter`
- 把图片序列写成 `mp4`

#### 5.5.4 官方 `examples/sa_pc.csv` 在 H20 上不可用

后续想直接用官方 example 做 sanity check 时，又发现：

- `examples/sa_pc.csv` 在当前 checkout 里实际上不存在

因此最终采用的是：

- 自行构造单样本 CSV
- 再跑单样本 `SA` / `PC` 诊断

这也说明第三方 repo 的 README、examples 与当前实际 checkout 并不总是完全一致，不能盲信。

## 6. 从“全 0”到“全 1”再到“全空”的演化

### 6.1 阶段一：全 0

第一次大规模跑通 `CS:GO val` 后，得到的结果是：

- `SA = 0.0`
- `PC = 0.0`
- `joint = 0.0`

但进一步检查日志发现，这并不是有效打分，而是：

- 模型输出空串 `''`
- parser 无法解析
- 最后被默认兜底成 `0`

典型日志表现：

- `Warning: Could not parse output ''. Defaulting to 0.`

因此：

- 第一阶段的全 `0` 不是“模型真的觉得所有视频都很差”
- 而是“模型没有产生可解析答案，parser 把失败输出写成了 0”

### 6.2 阶段二：非空输出但塌缩成 1

后续为了让模型“至少能输出点东西”，我们对官方 inference 路径做了一系列修改：

- `3efb1ca` Fix videophy generation defaults
- `eda14f7` Use greedy decoding for videophy eval
- `2903d6b` Tighten videophy score parsing

这一阶段出现的典型原始输出包括：

- `'100000'`
- `'1.1.1.'`
- `'1'`

在当时的宽松 parser 下，它们大多都会被压成：

- `1`

于是就出现了第二阶段现象：

- `CS:GO val` 405 个样本，`SA` 全是 `1.0`
- `PC` 全是 `1.0`

对应 summary：

- `sa_mean = 1.0`
- `pc_mean = 1.0`
- `joint = 0.0`
- `count = 405`

这说明：

- 系统不再是空输出
- 但又塌成了“常量低分输出器”

### 6.3 阶段三：严格解析后暴露为“坏输出”

根据 review 发现，我们又把 inference 路径重新收了一版，commit：

- `8c7b60a` Harden VideoPhy score decoding

这次主要做了三件事：

1. 不再解码“最后 `max_new_tokens` 个 token”的 tail，而是解码完整生成结果
2. 把 `processor.tokens_to_generate` 和 `max_new_tokens` 解耦
3. 引入严格 parser，只接受“唯一、合法”的分数

这轮之后，单样本在 H20 上的输出变成：

- `SA raw_output = '100000'`
- `PC raw_output = '100000'`

严格 parser 的结果是：

- `score` 留空

这一步很关键，因为它证明了：

- 之前的“全 1”至少部分是 parser 塌缩伪装出来的
- 模型本身当前更像是在稳定地产生一种坏串，而不是有效分数

### 6.4 阶段四：根因确认与最终修复

最后把 H20 上“为什么几乎全是 `1`”这件事彻底解释清楚后，可以把根因归纳为三层叠加：

1. **旧 inference 没有稳定只看“模型真正新生成的答案 token”**
   - 早期实现混用了完整输出、尾部 token 和 prompt 附带内容
   - 这会把并不是真正评分答案的文本也拿去解析

2. **生成长度过长，容易产生坏串**
   - 典型坏输出包括：
     - `'100000'`
     - `'1.1.1.'`
     - `'1'`
   - 这些字符串并不等于“模型认真给了 1 分”

3. **旧 parser 过于宽松，会把坏串错误折叠成合法分数 `1`**
   - 例如 `'100000'`、`'1.1.1.'` 这类串，只要里面第一个合法 digit 是 `1`
   - 最终就会被 summary 误记成 `1`

所以最开始文档里记录到的“很多样本几乎都输出一样的数字 `1`”，其本质不是：

- judge 认真地对所有视频都做出了同样的低分判断

而更接近于：

- **坏输出被旧解析逻辑错误折叠成了常量 `1`**

最终的修复方案包括：

- 只解码真正新生成的 token，不再混读 prompt/tail
- 收紧 `max_new_tokens`，避免生成无意义长尾
- `attention_mask` 固定使用整型
- 优先解析 assistant 的真实回答文本
- 文本解析失败时，回退到合法评分 token 的首步 logits 选分
- H20 默认改成 `fp32`，绕开 `bf16` 路径导致的 `SIGFPE`
- 单个坏视频按行跳过，不再让整 shard 失败
- 并行脚本支持启动前自动清理目标 GPU 上的旧进程

修复完成后，我们已经在 H20 上拿到非塌缩的有效分数，后文第 10 节给出最终结果。

## 7. 额外 sanity check：自然视频三样本也塌成 1

为了排除“是不是 CS:GO 数据集本身太特殊”，我们额外测试了三段自然视频帧序列：

- `bear`
- `bike-packing`
- `blackswan`

这些视频首先被用 `cv2.VideoWriter` 转成 `mp4`，然后送进同一套 VideoPhy-2-AutoEval。

得到的结果是：

- `SA`：三段全是 `1.0`
- `PC`：三段全是 `1.0`

这说明问题并不只是：

- “CS:GO 是游戏视频所以 OOD”

更像是：

- 当前 H20 上这套 `videophy_2_auto + 现有 inference path` 整体已经退化

## 8. 当前 code review 结论

根据后续 review 和本地代码对照，目前已经确认以下高风险点。

### 8.1 生成答案与 tail token 解码混在一起

当时的推理链中：

- `processor.tokens_to_generate` 被直接绑到 `max_new_tokens`
- `processing_mplug_owl.py` 会先在 prompt 后面补 `eos_token_id`
- `inference.py` 又只去解码最后 `max_new_tokens` 个 token

这会导致：

- 解码内容并不一定是真正“模型新生成的答案”
- 很容易读到 padding / tail artifact

### 8.2 宽松 parser 会把很多坏输出都压成 1

例如：

- `'100000'`
- `'1.1.1.'`
- `'1'`

在宽松 parser 下都会被读成 `1`。

这意味着：

- “全 1”并不等于“模型认真做了低分判断”
- 其中相当一部分只是坏输出被错误折叠

### 8.3 greedy + single-digit prompt 会把系统推向低熵常量输出

前一阶段修复里有两类变化会同向放大塌缩：

- 默认 greedy
- prompt 强制 “single digit only”

对生成式 evaluator 来说，如果视频条件分支信号弱、domain mismatch 或上下文边界有问题，就很容易退化成：

- 始终吐一个低熵短 token

### 8.4 wrapper 不是主因

当前证据看：

- manifest -> videopath/caption 的映射是正常的
- summary 逻辑本身也不是“全 1”主因

所以问题焦点仍然在：

- `third_party/videophy/VIDEOPHY2/inference.py`

而不是：

- summary 层
- manifest 层
- VideoPhy wrapper 层

## 9. 我们已经做过的修复

截至目前，我们已经做过的 VideoPhy-2 相关修补包括：

- `7e6f83d`：
  - 稳定 H20 上的 temporal conv hack
- `e5e0727`：
  - 允许通过环境变量覆盖 dtype
- `4cf4498`：
  - 提升 launcher 的中间日志和 GPU 清理能力
- `3efb1ca`：
  - 调整生成默认值
- `eda14f7`：
  - 默认改用 greedy
- `2903d6b`：
  - 调整 parser 和 prompt
- `8c7b60a`：
  - 重新收紧 score decoding
  - 引入严格 parser
  - 不再把坏输出硬压成有效分数
- `32cb9e4`：
  - 只解析真正 assistant 输出的评分答案
  - 只看真正新生成 token
  - 加入首步 logits 的合法评分 fallback
- `397e5ac`：
  - 去掉额外前向，直接复用 `generate(..., output_scores=True)`
  - 降低 H20 上的峰值显存
- `2a764fd`：
  - H20 默认使用 `fp32`
  - 修复 `bf16` 路径下的 `SIGFPE`
- `3a705e9`：
  - 单个坏视频只记 `error` 并跳过
  - 不再因为某一条生成视频坏掉而中断整个 shard

另外，在最终使用阶段我们又对 launcher 做了一次收口：

- 新增静默运行模式，避免中间刷屏
- 新增双模型最终汇总脚本，只在 `LingBot-base` 和 `LingBot-Stage1` 都完成后打印两张表

## 10. 最新 H20 最终结论

### 10.1 当前 VideoPhy-2 链路已经恢复可用

截至 `2026-04-12` 的最终验证结果，H20 上的这条 `VideoPhy-2-AutoEval` 链路已经恢复稳定可用：

- `CS:GO test` 数据集直测：成功
- `LingBot-base test_inf_result`：成功
- `LingBot-Stage1 test_inf_result`：成功

并且打分已经不再塌缩成常量 `1`，而是能产生有区分度的均值结果。

### 10.2 最终跑通的 H20 结果

我们最终在 H20 上得到的 aggregate summary 如下：

- `dataset_test`
  - `SA Mean = 3.95`
  - `PC Mean = 3.5375`
  - `Joint >= 4 = 0.4375`
- `lingbotbase`
  - `SA Mean = 4.1549`
  - `PC Mean = 3.5493`
  - `Joint >= 4 = 0.4789`
- `lingbotstage1`
  - `SA Mean = 4.2329`
  - `PC Mean = 3.6027`
  - `Joint >= 4 = 0.5205`

这说明：

- 现在的 judge 已经不再是“常量 `1` 输出器”
- 也不再是“空输出 -> 假 `0`”的失败状态
- 对不同模型和不同 shard 已经能给出非塌缩的有效分数

### 10.3 当前最需要注意的工程条件

现在这条链路虽然已经可用，但 H20 上还有一个非常现实的工程前提：

- **目标 GPU 必须是干净的**

实际运行中，如果 `0-7` 卡上本来就挂着旧的 `python` 进程，最容易出现的现象是：

- 某些 shard 无故 `exit 1`
- 同一批视频这次过、下次不过
- 但代码和数据本身并没有问题

因此，当前最稳的运行方式是：

- 启动前先清卡
- 或直接在脚本里设置 `KILL_EXISTING_GPU_PIDS=1`

## 11. VideoPhy-2 benchmark 与 AutoEval 的关系

必须明确区分：

- `VideoPhy-2 benchmark`
- `VideoPhy-2-AutoEval`

二者关系是：

- benchmark 定义任务、prompt、评测设定
- AutoEval 只是一个自动 judge

所以：

- 只拿你自己的视频喂 AutoEval
- 不按官方 benchmark 输入协议生成

这并不等于“完成了标准 VideoPhy-2 benchmark”

标准 benchmark 的核心逻辑应该是：

1. 用 VideoPhy-2 官方 prompt 跑模型
2. 生成视频
3. 用人工评测或可靠的 AutoEval 打分
4. 报 `SA` / `PC` / `joint`

## 12. LingBot / LingBot-Stage1 为什么更复杂

### 12.1 LingBot 不是纯 prompt-only T2V

我们当前的 Stage-1 训练/推理堆栈并不只吃文本 prompt。

从代码看，样本中还包含：

- `video`
- `poses`
- `actions`
- `intrinsics`

见：

- [stage1_components.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py#L77)

而 helper 在推理侧又会构建：

- prompt 的 T5 context
- 从视频首帧构造 first-frame latent
- 从 camera poses / actions / intrinsics 构造 control signal

见：

- [stage1_components.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py#L187)
- [stage1_components.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py#L197)
- [stage1_components.py](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py#L213)

### 12.2 这意味着必须分两条 protocol

因此，LingBot / LingBot-Stage1 的评测必须分开：

- **Track A：标准 benchmark track**
  - 只允许 benchmark-safe 输入
- **Track B：内部 conditioned track**
  - 允许额外控制、first-frame、trajectory 等

如果继续使用：

- GT first frame
- GT poses
- GT actions
- GT intrinsics

那它就不能被称为：

- “标准 VideoPhy-2 benchmark result”

最多只能写成：

- “conditioned internal VideoPhy-style evaluation”

## 13. LingBot / LingBot-Stage1 的正确 VideoPhy-2 protocol

这部分的详细说明已经单独写在：

- [PROTOCOL_lingbot_videophy2_standard_eval.md](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/PROTOCOL_lingbot_videophy2_standard_eval.md)

这里给出压缩版结论。

### 13.1 如果要声称“在 VideoPhy-2 上评测”

必须满足：

1. 使用官方 VideoPhy-2 prompt split
2. LingBot-base 和 LingBot-Stage1 在相同 prompt / seed / 生成超参下生成
3. 不使用 GT-derived oracle 条件
4. 用同一 judge 打分
5. 报 `SA`、`PC`、`joint`

### 13.2 如果模型必须吃更多条件

如果 LingBot 当前必须依赖：

- first frame
- poses
- actions
- intrinsics

那就不能直接写成标准 benchmark 结果。

这时正确表述是：

- “内部 conditioned 评测”
- “在 VideoPhy-2 prompt 上的 conditioned evaluation”

而不是：

- “official VideoPhy-2 benchmark result”

## 14. 当前 repo 中最接近正式 protocol 的入口

如果只看脚本组织方式，最接近的是：

- [run_videophy2_lingbot_parallel.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_videophy2_lingbot_parallel.sh)

它已经默认比较：

- `exp_base_zeroshot`
- `exp_stage1_epoch2`

并且跨三个 seed：

- `42`
- `123`
- `3407`

但它当前的默认 manifest 是：

- `data/manifests/csgo_phys_val50.csv`

见：

- [run_videophy2_lingbot_parallel.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_videophy2_lingbot_parallel.sh#L20)

所以它现在更像：

- 一个方便的内部评测 wrapper

还不是：

- 严格意义上的 VideoPhy-2 官方 benchmark runner

## 15. 当前推荐写法

如果要把当前阶段写进报告或论文内部记录，建议使用下面这段更新后的结论。

### 15.1 可直接引用的中文结论

在 H20 环境中，我们完成了 `VideoPhy-2-AutoEval` 所需的权重部署、依赖修复、dtype 稳定化、显存收敛、输出解析修复、坏样本容错以及并行脚本 GPU 清理能力补强。最早阶段出现的“全 `0`”来自空输出解析失败，后续阶段出现的“几乎全 `1`”则来自坏输出（如 `'100000'`、`'1.1.1.'`）被旧解析逻辑错误折叠成合法分数 `1`。在修复生成解码、收紧 parser、加入 logits fallback、改用 `fp32` 并避免脏 GPU 干扰之后，VideoPhy-2-AutoEval 已经能够在 H20 上稳定跑完 `CS:GO test`、`LingBot-base test_inf_result` 和 `LingBot-Stage1 test_inf_result`，并产生非塌缩、可区分的 `SA/PC/joint` 指标。需要单独说明的是，LingBot / LingBot-Stage1 当前仍然不是严格的 prompt-only benchmark-safe inference 形态，因此这些结果更适合表述为当前项目设定下的 VideoPhy-2-style AutoEval 结果，而不是直接声称与官方 leaderboard 完全可比的标准 benchmark 成绩。

## 16. 建议的下一步

### 16.1 继续保留 benchmark-safe 与 conditioned 两条口径

当前最合理的做法不是放弃 VideoPhy-2，而是把口径分清：

- `benchmark-safe`
  - 只允许 prompt 驱动
- `conditioned evaluation`
  - 允许现有 LingBot / Stage1 额外条件

### 16.2 工程运行方式建议

H20 上建议固定使用：

- 启动前清空目标 GPU
- 或在脚本里设置 `KILL_EXISTING_GPU_PIDS=1`

对于 `test_inf_result` 的最终展示，建议直接使用新的双模型脚本：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
KILL_EXISTING_GPU_PIDS=1 bash scripts/run_videophy2_test_inf_result_dual_summary.sh
```

这个脚本会：

- 先顺序跑完 `lingbotbase` 与 `lingbotstage1`
- 中间不再刷大量 shard 日志
- 最后只打印两张 summary 表

### 16.3 后续研究建议

后续如果要进一步提升可比性，仍然建议：

- 设计更接近官方 benchmark-safe 的 LingBot inference mode
- 保留人工抽检
- 与其他物理一致性指标交叉验证

## 17. 目前关键 commit 列表

截至目前，和这条问题链直接相关的 commit 包括：

- `89756a0` Add H20-2 path defaults and evaluation launchers
- `48dbbb6` Set default H20-2 LingBot code paths
- `7e6f83d` Stabilize VideoPhy CPU temporal conv on H20
- `e5e0727` Allow VideoPhy inference dtype override
- `4cf4498` Improve VideoPhy launcher visibility and GPU cleanup
- `3efb1ca` Fix videophy generation defaults
- `eda14f7` Use greedy decoding for videophy eval
- `2903d6b` Tighten videophy score parsing
- `8c7b60a` Harden VideoPhy score decoding
- `32cb9e4` Fix VideoPhy2 score decoding on test eval
- `397e5ac` Reduce VideoPhy2 inference peak memory
- `2a764fd` Default VideoPhy2 H20 runs to fp32
- `3a705e9` Skip bad VideoPhy2 rows instead of aborting shards

## 18. 最终一句话总结

截至 `2026-04-12`，`VideoPhy-2-AutoEval` 在 H20 上已经被修到可稳定运行且不再塌缩成常量 `1`；当前需要注意的主要不是“judge 本身坏掉”，而是运行时必须保证目标 GPU 干净，以及在论文表述中明确区分 `benchmark-safe` 与 `conditioned evaluation` 两条口径。

## 19. 切换到 Physics-IQ 的后续方案

在确认 `VideoPhy-2-AutoEval` 不可用之后，下一步更合理的路线是转向 `Physics-IQ`。

### 19.1 为什么 Physics-IQ 不能直接当作任意视频 judge

`Physics-IQ` 的官方协议不是“给一段视频直接打分”，而是：

- 给定官方 benchmark 场景
- 使用真实参考视频与真实运动 mask
- 将模型生成视频与参考视频进行逐帧与逐 mask 对比
- 最后汇总成 `Physics-IQ score`

因此，Physics-IQ 更接近：

- 基于参考视频的 paired metric

而不是：

- 类似 VideoPhy-2-AutoEval 那样的无参考语言 judge

### 19.2 面向 CSGO 的可执行方案

针对我们自己的 `CSGO val` 数据，更可行的用法是实现一个：

- `Physics-IQ-style paired evaluator`

其思路是：

- 把真实 `CSGO val` clip 当作 reference
- 把待测视频当作 candidate
- 复用 Physics-IQ 的核心时空一致性指标：
  - frame-wise MSE
  - spatiotemporal IoU
  - spatial IoU
  - weighted spatial IoU

这样：

- 先用 `real -> real` 跑一遍 sanity check
- 再用 `generated -> real` 比较 LingBot-base / LingBot-Stage1

### 19.3 口径必须写清楚

如果后续采用这条路线，报告里必须明确写成：

- `Physics-IQ-style paired evaluation on CSGO`

而不是：

- `official Physics-IQ benchmark result`

原因是：

- 我们没有使用 Physics-IQ 官方 198 个测试场景
- 也没有使用 Physics-IQ 官方真实参考集与官方 mask 目录结构

所以它是：

- 受 Physics-IQ 启发的内部 paired metric

而不是：

- 官方 leaderboard 可比结果

### 19.4 当前代码改造方向

后续接入代码时，建议增加：

- `src/physical_consistency/eval/physics_iq.py`
- `src/physical_consistency/cli/run_physics_iq.py`
- `scripts/run_physics_iq_dataset_parallel.sh`

默认先支持：

- `reference = real CSGO val clip`
- `candidate = real CSGO val clip`

用来验证：

- 整条 Physics-IQ-style metric 链路本身是正常的

之后再切换到：

- `candidate = generated videos`
- `reference = matched real val clips`

以比较 LingBot-base / LingBot-Stage1。

### 19.5 H20 上已完成的 `Physics-IQ-style real-vs-real` 单样本 sanity check

在完成工作区清理、确认 `world_model_phys` 主线 repo 干净之后，我们已经在 H20 上成功跑通了一条单样本 `real-vs-real` sanity check。

使用命令：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
conda activate /home/nvme03/workspace/world_model_phys/.conda_envs/phys-videophy

python - <<'PY'
import pandas as pd
src = "/home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/metadata_val.csv"
dst = "/tmp/csgo_val_one.csv"
pd.read_csv(src).head(1).to_csv(dst, index=False)
print(dst)
PY

PYTHONPATH=src python -m physical_consistency.cli.run_physics_iq \
  --config /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/configs/physics_iq_dataset_eval.yaml \
  --experiment_name exp_dataset_val_physics_iq_real_one \
  --manifest_csv /tmp/csgo_val_one.csv \
  --reference_source_root /home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3 \
  --candidate_source_root /home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3 \
  --reference_source_mode dataset_clip \
  --candidate_source_mode dataset_clip \
  --output_root /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys \
  --seed 0
```

H20 上的单样本输出为：

- `compare_frame_count = 40.0`
- `mse_mean = 0.0`
- `spatiotemporal_iou_mean = 1.0`
- `spatial_iou = 1.0`
- `weighted_spatial_iou = 1.0`
- `physics_iq_style_score = 100.0`

这说明：

- `Physics-IQ-style paired evaluator` 在 H20 上已经能够稳定工作
- `real -> real` 的 identity sanity check 与预期一致
- 相比之下，当前真正的剩余问题已经不在 evaluator 本身，而在“如何拿到干净、可复现的 LingBot candidate 视频”

### 19.6 `LingBot-base vs real` 单样本评分的当前状态

截至目前，`LingBot-base vs real` 单样本评分 **尚未在 H20 上完成**，原因不是 `Physics-IQ-style` evaluator 有问题，而是：

- `LingBot-base` 生成依赖的外部 `code/` 目录并不属于 `world_model_phys` 主线 repo
- 该目录最初只是合作者提供的额外代码快照，并不受当前主线 git 管理
- 在后续调试过程中，这份外部代码被单独修改、误清空和最终移除，以恢复 H20 上主线工作区的干净状态

因此，当前最准确的状态是：

- `Physics-IQ-style real-vs-real`：已经通过单样本 sanity check
- `LingBot-base / LingBot-Stage1`：H20 上的 generation-only 流程已经重新接通，并且已经能够稳定产出 candidate videos

### 19.7 当前 repo 已支持“直接给两条视频路径”的单样本 Physics-IQ 评分

为了避免后续再依赖整套外部 wrapper，当前 `Physics-IQ-style` CLI 已经支持直接输入：

- `reference_videopath`
- `candidate_videopath`

也就是说，只要未来拿到一条 `LingBot-base` 生成视频，就可以不再构造完整 manifest，而是直接对单对视频评分。

示例命令如下：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
conda activate /home/nvme03/workspace/world_model_phys/.conda_envs/phys-videophy

PYTHONPATH=src python -m physical_consistency.cli.run_physics_iq \
  --config /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/configs/physics_iq_dataset_eval.yaml \
  --experiment_name exp_lingbot_base_vs_real_one \
  --reference_videopath /home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/val/clips/Ep_000028_team_2_player_0001_inst_000_clip0000/video.mp4 \
  --candidate_videopath /path/to/lingbot_base_candidate.mp4 \
  --sample_id Ep_000028_team_2_player_0001_inst_000_clip0000 \
  --clip_path val/clips/Ep_000028_team_2_player_0001_inst_000_clip0000 \
  --prompt "First-person view of a competitive CS:GO match on de_dust2. The player is moving through the map holding a Glock-18. Photorealistic game rendering with detailed textures, lighting effects, and HUD elements visible." \
  --output_root /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys \
  --seed 0
```

结果查看路径：

```bash
cat /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/runs/eval/physics_iq/exp_lingbot_base_vs_real_one/summary.json
cat /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/runs/eval/physics_iq/exp_lingbot_base_vs_real_one/seed_0/output_pairs.csv
```

这意味着后续只要指定一条已经生成好的 candidate video，`LingBot-base vs real` 或 `LingBot-Stage1 vs real` 的单样本 Physics-IQ 评分就可以立即执行，而不必再依赖之前那套混乱的外部 `code` 工作区。

### 19.8 H20 上已经完成 `metadata_test.csv` 的 80 条生成视频落盘

在完成 H20 环境整理、`flash_attn` 安装、`test` 子集构建以及 generation-only pipeline 修复之后，我们已经在：

- `/home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/metadata_test.csv`

上成功跑通了 `LingBot-base` 和 `LingBot-Stage1` 的 `80 / 80` 生成任务。

当前生成结果中最重要的几类文件如下。

| 文件/模式 | 含义 |
| --- | --- |
| `lingbotbase/videos/*.mp4` | `LingBot-base` 在 `metadata_test.csv` 上生成的 80 条视频 |
| `lingbotstage1/videos/*.mp4` | `LingBot-Stage1` 在 `metadata_test.csv` 上生成的 80 条视频 |
| `generated_videos.csv` | 生成视频和参考 clip 的对应清单 |
| `run_manifest.csv` | 本次运行使用的 manifest |
| `worker_manifests/*.csv` | 每张卡分到的 10 条子集 |

对应的 H20 绝对路径可以写成：

- `LingBot-base` 视频目录：
  - `/home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/test_inf_result/lingbotbase/videos`
- `LingBot-Stage1` 视频目录：
  - `/home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/test_inf_result/lingbotstage1/videos`
- `LingBot-base` 清单目录：
  - `/home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/test_inf_result/lingbotbase`
- `LingBot-Stage1` 清单目录：
  - `/home/nvme03/workspace/world_model_phys/PHYS/Dataset/processed_csgo_v3/test_inf_result/lingbotstage1`

因此，截至目前，H20 上已经不再缺少 `LingBot-base` / `LingBot-Stage1` candidate videos；后续真正要做的是基于这些已落盘视频，继续进行人工抽查、`Physics-IQ-style` 后评测，或其他补充指标汇总。

### 19.9 H20 上最终 VideoPhy-2 命令与终端输出记录

为了让这份文档同时保留“最终成功跑通时到底在终端里看到了什么”，这里把关键命令和终端输出摘录如下。

#### 19.9.1 `dataset_test` 的最终 aggregate 输出

对应运行阶段中，`CS:GO test` 数据集直测最终打印出的 aggregate summary 为：

```text
VideoPhy-2 Summary: exp_dataset_test_autoeval_parallel

Overall
| Metric | Mean | Count |
| --- | --- | --- |
| SA Mean | 3.95 | 1 |
| PC Mean | 3.5375 | 1 |
| Joint >= 4 | 0.4375 | 1 |
```

#### 19.9.2 最终双模型静默汇总命令

H20 上最终用于只输出两张表的命令为：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
git pull origin main
KILL_EXISTING_GPU_PIDS=1 bash scripts/run_videophy2_test_inf_result_dual_summary.sh
```

#### 19.9.3 最终双模型静默汇总终端输出

H20 上这条命令最终打印出的结果如下：

```text
lingbotbase

Overall
| Metric | Mean | Count |
| --- | --- | --- |
| SA Mean | 4.1549 | 1 |
| PC Mean | 3.5493 | 1 |
| Joint >= 4 | 0.4789 | 1 |

lingbotstage1

Overall
| Metric | Mean | Count |
| --- | --- | --- |
| SA Mean | 4.2329 | 1 |
| PC Mean | 3.6027 | 1 |
| Joint >= 4 | 0.5205 | 1 |
```

#### 19.9.4 这些输出从 VideoPhy-2 角度意味着什么

从 `VideoPhy-2-AutoEval` 的 judge 视角看，这两张表可以解释为：

- `SA Mean`
  - `LingBot-base = 4.1549`
  - `LingBot-Stage1 = 4.2329`
  - 说明两者整体都已经能够较好地遵循文本语义，`Stage1` 略优于 `Base`
- `PC Mean`
  - `LingBot-base = 3.5493`
  - `LingBot-Stage1 = 3.6027`
  - 说明两者在物理常识一致性上处于中上水平，但物理性仍然弱于语义符合度
- `Joint >= 4`
  - `LingBot-base = 0.4789`
  - `LingBot-Stage1 = 0.5205`
  - 说明大约有 `48%` 与 `52%` 的视频，能够同时满足“语义上过关且物理上也过关”

因此，最终的 VideoPhy-2 结论可以简写为：

- `LingBot-Stage1` 相比 `LingBot-base` 有稳定但温和的提升
- 当前两者的主要短板都不是 `SA`，而是 `PC`
- 也就是说，模型在“视频内容是否符合描述”上做得比“视频运动和交互是否足够物理合理”更好

## 20. H20 上 Stage1/TRD 双 Student 训练排障与当前稳定方案

这一节用于单独记录我们在 H20 上启动 `Stage1 / TRD / dual` 训练时遇到的全部关键问题、修复过程和当前稳定做法。

### 20.1 当前训练目标与入口

当前训练入口为：

- [run_train_trd_v1_dual.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_train_trd_v1_dual.sh)
- [run_train_trd_v1.sh](/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_train_trd_v1.sh)

H20 上的实际启动目录为：

- `/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys`

当前常用命令为：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
git pull origin main

GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_train_trd_v1_dual.sh \
  --project_name intro-example \
  --wandb_entity WorldModel_11 \
  --num_frames 81
```

日志与 PID 文件位置为：

- 完整日志：
  - `/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_dual.log`
- PID 文件：
  - `/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_dual.pid`

重新登录后建议的监控命令为：

```bash
tail -n 200 -F /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_dual.log
```

### 20.2 Student、Teacher 与当前蒸馏链路

当前这套训练不是“VideoREPA 提供 student backbone，再由我们套壳训练”，而是：

- `student`：
  - LingBot / Wan 自带的 `WanModel`
- `teacher`：
  - VideoREPA 侧的 `VideoMAEv2Teacher`
- `distillation target`：
  - student 中间 block token 与 teacher 输出 token 的 relation distillation

更具体地说：

- student 主干来自 `LingBot/Wan`
- teacher 来自 `VideoREPA`
- 当前默认从 student 的 `student_target_block = 20` 提取中间特征
- loss 由两部分组成：
  - `loss_fm`
  - `loss_trd`
- 总 loss 为：
  - `loss_total = loss_fm + lambda_trd * loss_trd`

### 20.3 `epoch`、`micro_step`、`optimizer step` 到底是什么意思

当前配置中：

- `num_epochs = 5`
- `gradient_accumulation_steps = 4`

因此训练时有两套“步数”：

- `micro_step`
  - 每处理一个 micro-batch 就加 `1`
- `global_step`
  - 只有累计满 `4` 个 micro-step，真正执行一次 `optimizer.step()` 时才加 `1`

所以：

- `4 micro-steps = 1 optimizer step = 1 global_step`

H20 上当前训练计划日志会打印：

```text
[TRAIN PLAN] epochs=5 micro_steps_per_epoch=209 optimizer_steps_per_epoch=52 total_optimizer_steps=260 grad_accum=4 dataset_samples=1670 world_size=8
```

这意味着：

- 总共训练 `5` 个 epoch
- 每个 epoch 大约 `209` 个 micro-step
- 每个 epoch 大约 `52` 个真正参数更新 step
- 整个 run 总共大约 `260` 个 optimizer steps

### 20.4 训练开始阶段最早遇到的两个核心问题

#### 20.4.1 `LoRA + ZeRO-3 + block-level checkpointing` 的 metadata mismatch

最初在引入 LoRA 以解决显存压力之后，训练会在 backward 重计算时直接报错：

```text
torch.utils.checkpoint: Recomputed values for the following tensors have different metadata than during the forward pass
```

根因是：

- `LoRA`
- `ZeRO-3`
- 原始 block 级 `gradient checkpointing`

三者组合后，在 backward 重算时 tensor metadata 不一致，导致 `checkpoint` 直接失败。

第一阶段修复思路是：

- LoRA 模式下先暂时跳过 student block 级 checkpointing

对应修复 commit：

- `c034245` Skip block checkpointing for LoRA student path

#### 20.4.2 关闭 checkpointing 后立刻转成 OOM

LoRA 生效之后，`after_accelerator_prepare` 的显存明显下降，但训练真正开始时又会报 CUDA OOM。

典型表现为：

- `after_accelerator_prepare` 只有 `9.68 GiB`
- 但首个真实前向后，单卡会涨到 `88+ GiB`
- 最终在 `54 MiB` 或 `136 MiB` 的额外分配上爆掉

这不是 LoRA 失效，而是：

- LoRA 主要节省的是可训练参数、梯度、优化器状态
- 并不自动节省长时空序列的 activation
- 一旦关闭 block 级 checkpointing，`81` 帧长序列 activation 会直接成为新的显存主因

### 20.5 为什么 96G H20 之前还会 OOM

最初 OOM 阶段，用户最关心的问题是：

- H20 明明有 `96G`
- 已经用了 `LoRA + ZeRO-3 + 一堆 trick`
- 为什么还是会爆

当前已经确认的解释如下。

#### 20.5.1 `81` 帧下的 student 序列长度

对于早期的 `81` 帧配置，student 的时空几何大致是：

- `num_frames = 81`
- latent grid:
  - `T = 21`
  - `H = 60`
  - `W = 104`
- patch size:
  - `(1, 2, 2)`
- 最终序列长度：
  - `seq_len = 21 * 30 * 52 = 32760`

#### 20.5.2 激活量级的直观估算

以 hidden size 约 `5120`、`bf16` 为例，一个最基础的主激活张量：

- `[1, 32760, 5120]`

其大小约为：

- `32760 * 5120 * 2 bytes ≈ 0.312 GiB`

也就是说，仅一个主 `[B, S, D]` 级别张量就有三百多 MiB。

而早期日志中反复出现的两个关键数字：

- `54 MiB`
- `136 MiB`

也是能精确对应上的：

- `54 MiB`
  - 约对应 `student_ffn_chunk_size = 2048` 时，FFN 中间 chunk `[1, 2048, 13824]`
- `136 MiB`
  - 约对应 FFN 基座线性层一个 `bf16` 权重矩阵 `[13824, 5120]`

因此，早期 OOM 的本质不是“LoRA 没起作用”，而是：

- `long-sequence activation + ZeRO-3 参数 gather 峰值`

### 20.6 中间加过的所有排障与省显存 trick

这一路实际加过的关键手段如下。

#### 20.6.1 显存与序列诊断日志

为了先看清到底在哪一步炸，我们先后加入了：

- `[SEQ GEOM]`
  - 直接打印 `num_frames -> latent_grid -> seq_len`
- `[GPU MEM]`
  - `after_low_model_load`
  - `after_high_model_load`
  - `before_accelerator_prepare`
  - `after_accelerator_prepare`
  - `after_teacher_load`
  - `after_teacher_encode`
  - `step_*_start / after_forward / after_backward / after_optimizer_step / after_zero_grad`

对应相关 commit：

- `143d819` Add sequence geometry and per-step memory logs

#### 20.6.2 减小 `student_ffn_chunk_size`

为了减小 FFN 前向峰值，我们将：

- `student_ffn_chunk_size: 2048 -> 512`

这主要影响速度和单次前向分块，不是训练目标本身的改动。

#### 20.6.3 重新开启兼容 LoRA 的 checkpointing

后续为了把 activation 再压下去，我们没有回到原始那种会报 metadata mismatch 的路径，而是改成了更兼容的 reentrant checkpointing 方案。

对应相关 commit：

- `21213a6` Re-enable student checkpointing in a more compatible path

#### 20.6.4 ZeRO-3 与 CPU param offload

当前稳定路径中保留了：

- `ZeRO-3`
- `CPU param offload`

注意：

- 没有重新开启 `optimizer offload`
- 因为之前它与当前 CUDA / torch / deepspeed 环境存在 CPUAdam 构建冲突风险

#### 20.6.5 启动前强制清卡

当前训练脚本会在启动前：

- 检查目标 GPU 上已有 compute 进程
- 直接强制 kill 掉这些进程

相关日志形式为：

```text
[GPU RESET] Force killing existing compute PIDs on GPUs ...
```

这一步的目的不是“温柔恢复训练”，而是：

- 强制结束旧训练
- 释放 GPU 显存
- 再启动新的 run

### 20.7 `70` 帧与 `69` 帧问题：为什么后来默认变成了 `69`

在前一轮省显存实验里，我们曾经尝试将帧数降低到 `70` 帧。

结果出现了新的 shape 报错：

```text
shape '[1, 18, 4, 60, 104]' is invalid for input of size ...
```

根因是：

- 旧版 `prepare_y()` 中一条 mask / temporal reshape 逻辑默认假设原始帧数满足 `1 + 4k`
- `81 = 1 + 4 * 20`
- `69 = 1 + 4 * 17`
- `70` 不满足这个模式

因此后来默认帧数先改成了：

- `69`

对应相关 commit：

- `7e90c33` Make temporal mask construction robust
- `28a0896` Default to `69` frames to satisfy `1 + 4k`

### 20.8 这里的 `mask` 到底是什么

这里的 `mask` 不是常见图像 inpainting 里的“黑色遮挡区域”。

它更准确地说是一个：

- `temporal conditioning mask`

含义是：

- 第一个时间步对应的 latent 条件是已知的
- 后面时间步的条件是未知的

也就是说，它表达的是：

- “首帧我看见了”
- “后面帧我没看见，你自己往后生成”

它不是：

- 在原图上遮一块黑色区域让模型补洞

而是：

- 在时间轴上指明“哪一帧是已知条件，哪一帧需要模型继续外推”

### 20.9 后续遇到的运行期 / 日志期错误

在显存问题被压住之后，后续报错主要不再是训练主体问题，而是日志、可视化和 CLI 兼容问题。

按时间顺序，主要有以下几类。

#### 20.9.1 `grad_norm is None`

报错形式：

```text
float() argument must be a string or a real number, not 'NoneType'
```

根因：

- 某些 deepspeed 路径下 `clip_grad_norm_()` 可能返回 `None`
- 日志代码里直接 `float(None)` 导致崩溃

对应修复 commit：

- `845b717` Guard `grad_norm=None`

#### 20.9.2 `bf16 tensor -> numpy` 失败

报错形式：

```text
Got unsupported ScalarType BFloat16
```

根因：

- relation matrix 可视化时直接对 `cpu + bf16` tensor 调用 `.numpy()`

对应修复 commit：

- `c11aa54` Convert relation matrices to `float32` before NumPy

#### 20.9.3 `wandb.Image(BytesIO)` 失败

报错形式：

```text
'_io.BytesIO' object has no attribute 'ndim'
```

根因：

- 当前环境下的 wandb 版本不接受 `BytesIO` 直接作为 image data

对应修复 commit：

- `2fe9b0e` Convert matplotlib output to NumPy RGB before `wandb.Image`

#### 20.9.4 `--num_frames 81` 一开始不能从 CLI 覆盖

报错形式：

```text
error: unrecognized arguments: --num_frames 81
```

根因：

- 训练 CLI 最初没有暴露 `--num_frames`

对应修复 commit：

- `5a0198a` Add `--num_frames` CLI override

### 20.10 W&B 面板爆炸、loss 丢失与后续整理

W&B 部分先后出现过三个独立问题。

#### 20.10.1 面板从几十张裂变成 `321` 张

根因：

- 早期把 `step_9_after_forward_*` 这种“带步号的 metric 名称”直接写进了 W&B
- W&B 会把每个新 metric 名都当作一个全新 panel

对应修复 commit：

- `25a9f46` Replace step-embedded metric names with stable progress/runtime metrics

#### 20.10.2 `setup_mem` 图表几乎没意义

这些图通常只有单个点，例如：

- `after_low_model_load`
- `after_accelerator_prepare`
- `after_teacher_load`

它们对排障有意义，但对日常盯训练几乎没有意义，因为：

- 不形成趋势
- 视觉噪音很大

后续处理方式是：

- 继续保留这些信息在日志里
- 不再把它们放进 W&B

对应修复 commit：

- `acdaf11` Keep setup memory in logs only

#### 20.10.3 训练 loss 明明在终端里变，但 W&B 里没有 `train/loss_*`

这也是最容易误导人的一个问题。

现象是：

- 终端 `[PROGRESS]` 明明在打印：
  - `loss_total`
  - `loss_fm`
  - `loss_trd`
- 但 W&B 里却搜不到 `train/loss_total`

根因已经定位清楚：

- `runtime/gpu_mem/*` 一度按 `micro_step` 作为 W&B 内部 `step`
- `train/*` 又按 `global_step` 作为 W&B 内部 `step`
- W&B 要求内部 `step` 单调递增
- 所以后写入的 `train/*` 被认为“step 倒退”，直接忽略

对应修复 commit：

- `558d888` Fix W&B train metric step ordering

必须强调：

- 这个修复只对 **新启动的 run** 生效
- 已经开始跑的旧 run 不会自动补回缺失的 `train/loss_*`

### 20.11 `nohup` 为什么还会被 SIGHUP 打断

我们在一次 81 帧训练中，明明已经用了 `nohup`，但日志最后仍然出现：

```text
Received Signals.SIGHUP death signal, shutting down workers
SignalException: Process ... got signal: 1
```

这不是：

- OOM
- 模型 forward / backward 出错
- 别的训练抢占 GPU

而是：

- `accelerate launch`
- `torch.distributed.run`
- `torch elastic`

这整条进程树没有真正完全脱离原始终端会话

因此即使外层 shell 看起来用了 `nohup`，只要：

- SSH 会话断开
- 或终端控制会话发出 `SIGHUP`

elastic 主进程仍可能收到 hangup 信号，然后主动把所有 worker 一起关掉。

也就是说：

- `nohup` 往往只能挡住一层
- 但不一定能完全挡住 `accelerate + elastic` 整条进程树

### 20.12 后台启动链路的最终加固

为了解决上面的 `SIGHUP` 问题，后台启动脚本后来经历了两轮加固。

#### 20.12.1 第一轮：`setsid + exec + </dev/null`

对应 commit：

- `b97e87a` Detach TRD launches from terminal session

#### 20.12.2 第二轮：Python launcher + `start_new_session=True`

最终更稳的版本不再只是把 `accelerate` 当作当前 shell 的后台 job，而是：

- 先启动一个很短命的 Python launcher
- 再由它派生真正训练进程

新派生出的训练进程会：

- `stdin = /dev/null`
- `stdout/stderr -> log file`
- `close_fds = True`
- `start_new_session = True`

对应 commit：

- `2143f60` Harden detached TRD launcher against SSH hangups

当前推荐理解是：

- 旧版 `nohup`：
  - 对普通单进程脚本通常够用
- 当前加固版 detached launcher：
  - 更适合 `accelerate + deepspeed + torch elastic` 这种多层训练栈

### 20.13 当前已经稳定跑起来的结论

截至目前，我们已经确认：

- 训练可以在 H20 上稳定进入真正的 optimizer step
- 不再卡在最早的 metadata mismatch
- 不再卡在最早的 `88+ GiB` OOM
- 不再卡在 `70` 帧的 shape mismatch
- 不再卡在 `grad_norm=None`
- 不再卡在 `bf16 -> numpy`
- 不再卡在 `wandb.Image(BytesIO)`
- 启动方式也已经加强到更抗 `SIGHUP`

当前在 `81` 帧下的典型日志如下：

```text
[TRAIN PLAN] epochs=5 micro_steps_per_epoch=209 optimizer_steps_per_epoch=52 total_optimizer_steps=260 grad_accum=4 dataset_samples=1670 world_size=8
[SEQ GEOM] num_frames=81 latent_grid=(21,60,104) patch_size=(1, 2, 2) seq_len=32760
[PROGRESS] epoch=1/5 global_step=19/260 micro_step=76 accum=4/4 loss_total=... loss_fm=... loss_trd=... peak_mem=... eta=...
```

在目前这条稳定路径下，`81` 帧训练的单步峰值显存大约为：

- `16.6 GiB ~ 19.2 GiB`

这和最开始的 OOM 阶段差很多，根因不是 H20 变了，而是训练策略已经改变为：

- LoRA
- ZeRO-3
- CPU param offload
- 兼容 LoRA 的 checkpointing
- 更小的 FFN chunk
- 额外的 memory-oriented patch

这些组合起来，已经把最初的 activation / parameter 双重显存压力一起拆掉了。

### 20.14 这些 trick 对精度的影响边界

当前加过的 trick 并不都一样。

#### 20.14.1 基本只影响速度 / 显存，不直接改变训练目标的

- `ZeRO-3`
- `CPU param offload`
- `gradient checkpointing`
- `student_ffn_chunk_size`
- `nohup / setsid / detached launcher`
- W&B / 日志重构

这些通常只改变：

- 显存占用
- 前向 / 反向分块方式
- 日志与可观测性

理论上不应显著改变最终优化目标。

#### 20.14.2 会真正影响训练上限或任务定义的

- `LoRA`
- 历史上把 `81` 帧改成 `69` 帧这件事

其中：

- `LoRA`
  - 把“全量微调”改成“低秩适配”
  - 更省显存，但最终容量上限可能低于 full fine-tune
- `81 -> 69`
  - 会改变时间范围与 temporal relation 建模范围

当前最新稳定实验已经重新把帧数切回：

- `81`

因此现在与最初原设定相比，最大的保守项主要还剩：

- `LoRA` 仍然存在

### 20.15 当前最推荐的终端监控方式

当前终端最有用的日志不是刷屏式进度条，而是这几类：

- `[TRAIN PLAN]`
- `[SEQ GEOM]`
- `[PROGRESS]`
- 关键阶段的 `[GPU MEM]`
- 错误与 traceback

其中：

- `[PROGRESS]` 已经等价于“文字版进度条”
- 会直接告诉你：
  - 当前第几个 epoch
  - 当前 `global_step / total_optimizer_steps`
  - 当前 `micro_step`
  - 当前 `loss_total / loss_fm / loss_trd`
  - 当前 `lr`
  - 当前 step 耗时
  - 当前 ETA

例如：

```text
[PROGRESS] epoch=1/5 global_step=19/260 micro_step=76 accum=4/4 loss_total=... lr=... peak_mem=... eta=11h45m
```

这条日志本身就已经能回答：

- “现在走到哪了”
- “loss 在不在更新”
- “还要多久跑完”
- “显存是否稳定”

### 20.16 当前最推荐的操作性结论

如果目标是：

- H20 上真正把 `Stage1 / TRD / dual` 稳定跑起来
- 并且尽可能接近最初想测的 `81` 帧设定

那么当前最推荐做法是：

1. 使用最新主线脚本启动
2. 用 `--num_frames 81` 明确覆盖
3. 启动后允许脚本在后台 detached 运行
4. 重新登录后用 `tail -F` 看日志
5. 把 `W&B train/loss_*` 的可视化检查放在“新启动且包含 `558d888` 之后的 run”上

推荐命令如下：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
git pull origin main

GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_train_trd_v1_dual.sh \
  --project_name intro-example \
  --wandb_entity WorldModel_11 \
  --num_frames 81
```

之后重新登录查看：

```bash
tail -n 200 -F /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_dual.log
```

如果要手动停掉当前 run，优先使用：

```bash
kill "$(cat /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_dual.pid)"
```

如果发现 PID 已不存在，再结合：

```bash
ps -ef | grep physical_consistency.cli.train_trd_v1 | grep -v grep
nvidia-smi
```

判断是否其实已经退出。

### 20.17 这一轮 H20 训练问题对应的关键 commit 列表

本轮 `Stage1 / TRD / dual` 训练排障过程中，最关键的提交包括：

- `c034245`
  - LoRA 模式下先跳过 student block checkpointing，绕开 metadata mismatch
- `143d819`
  - 增加 `[SEQ GEOM]`、per-step `[GPU MEM]`、降低 `student_ffn_chunk_size`
- `21213a6`
  - 恢复更兼容的 student checkpointing、保留 ZeRO-3 与 CPU param offload、启动前清卡
- `7e90c33`
  - 让 temporal mask / `prepare_y()` 对边界帧数更稳
- `28a0896`
  - 默认切到 `69` 帧以满足 `1 + 4k`
- `845b717`
  - 修复 `grad_norm=None`
- `c11aa54`
  - 修复 `bf16` relation matrix 转 `numpy`
- `2f7c362`
  - 引入 `nohup + full log + tail` 的启动方式
- `2fe9b0e`
  - 修复 `wandb.Image(BytesIO)` 问题
- `25a9f46`
  - 引入 `[TRAIN PLAN]` / `[PROGRESS]`，重构 runtime/progress 监控
- `5a0198a`
  - 增加 `--num_frames` CLI 覆盖
- `acdaf11`
  - `setup_mem` 仅写日志，不再污染 W&B
- `558d888`
  - 修复 `train/loss_*` 因 W&B step 冲突而丢失的问题
- `b97e87a`
  - 第一轮：让训练尽量脱离终端 session
- `2143f60`
  - 第二轮：使用 Python launcher + `start_new_session=True` 强化抗 `SIGHUP`

### 20.18 最终一句话总结

这轮 H20 训练链路目前已经从“最初几乎必炸的 OOM / checkpoint / wandb / SIGHUP 混合状态”，收敛到了：

- `81` 帧可启动
- 多卡 `dual` 训练可进入稳定 optimizer step
- 终端日志可直接看到 loss、步数、ETA、显存
- 后台 detached 启动方式更稳

当前真正需要持续关注的，已经不再是“这套链路能不能跑”，而是：

- `LoRA` 路线下最终精度上限如何
- `train/loss_*` 与后续验证结果是否收敛稳定
- 是否要进一步回到更接近 full fine-tune 的原始训练设定

## 21. V-JEPA 2.1 官方 Teacher 替换方案

### 21.1 变更目标

在不破坏当前已经稳定跑通的 `Stage1 / TRD / dual` 主训练骨架前提下，将 frozen teacher 从 `VideoMAEv2` 切换为 **官方 V-JEPA 2.1**。

约束：

- 必须使用官方代码
- 必须使用官方权重
- 不直接 `git clone` 官方 repo 到 `third_party`，避免嵌套 git
- 仍然保留现有 `TeacherEncoder -> TeacherFeatures -> TRD loss` 框架
- 验证从“按 step 触发”改成“每个 epoch 结束触发一次”

### 21.2 采用的官方来源

代码：

- 官方 GitHub：`https://github.com/facebookresearch/vjepa2`

权重：

- 官方公开权重下载域：`https://dl.fbaipublicfiles.com/vjepa2/`

当前默认采用的 teacher 变体：

- `vjepa2_1_vit_base_384`
- 对应权重：`vjepa2_1_vitb_dist_vitG_384.pt`

选择这个 base 版本的原因：

- 官方
- `feature_dim = 768`，能直接兼容现有 projector / `teacher_feature_dim`
- 当前环境没有 `xformers`，base 版本风险最低

### 21.3 代码改动概览

新增：

- `src/physical_consistency/teachers/vjepa2.py`
  - 新增 `VJEPA21Teacher`
  - 使用官方 `src/hub/backbones.py`
  - 只加载 encoder，不改动现有 TRD teacher 接口

修改：

- `src/physical_consistency/trainers/trd_v1.py`
  - 新增 `teacher_backend` 分发
  - `teacher_backend=vjepa2` 时实例化 `VJEPA21Teacher`
  - 支持 `.pt` 权重解析
  - 训练改为每个 epoch 结束触发验证
  - `optimizer_steps_per_epoch` 改为按 `ceil` 计算，更贴近真实更新次数

- `src/physical_consistency/trainers/stage1_components.py`
  - `compute_scheduler_total_steps()` 改为按 `ceil` 计算，避免调度器步数比真实 optimizer step 少

- `configs/train_trd_v1.yaml`
  - teacher 默认从 `VideoMAEv2` 切到 `V-JEPA 2.1`
  - 默认 `num_frames` 保持为当前稳定跑通的 `81`
  - 验证改为 `validation_every_epochs: 1`

- `scripts/fetch_vjepa2_official.sh`
  - 下载官方 source archive 到 `third_party/vjepa2_official`
  - 下载官方 checkpoint 到 `../weight/vjepa2_1`
  - 无嵌套 `.git`

### 21.4 当前默认路径

相对于 `world_model_phys` 项目根目录：

- teacher 代码目录：
  - `third_party/vjepa2_official`

- teacher 权重目录：
  - `../weight/vjepa2_1`

在 H20 上对应的目标路径就是：

- 代码：
  - `/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/third_party/vjepa2_official`

- 权重：
  - `/home/nvme03/workspace/world_model_phys/PHYS/weight/vjepa2_1`

### 21.5 当前默认 teacher 配置

- `teacher_backend: vjepa2`
- `teacher_model_variant: vjepa2_1_vit_base_384`
- `teacher_checkpoint_dir: ../weight/vjepa2_1`
- `teacher_image_size: 384`
- `teacher_input_frames: 64`
- `teacher_drop_first_frame: false`
- `teacher_feature_dim: 768`

对应关系是：

- student 仍然吃当前训练视频帧数（当前默认 `81`）
- teacher 会从这段视频里均匀采样 `64` 帧
- teacher 输出 token 后，TRD loss 仍通过现有 `_match_time()` 自动对齐时间维

### 21.6 验证触发方式

之前：

- `validation_every_steps: 300`

现在：

- `validation_every_steps: 0`
- `validation_every_epochs: 1`

也就是：

- 不再按 step 触发验证
- 每个 epoch 结束后触发一次验证流程

### 21.7 H20 上的推荐执行顺序

先停当前旧 teacher 训练：

```bash
kill "$(cat /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_dual.pid)" 2>/dev/null || true
pkill -u "$USER" -f "physical_consistency.cli.train_trd_v1" 2>/dev/null || true
nvidia-smi
```

拉最新代码：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
git pull origin main
```

下载官方 V-JEPA 2.1 代码与权重：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys
bash scripts/fetch_vjepa2_official.sh
```

如外网不稳，可先尝试：

```bash
bash /home/nvme01/clash-for-linux/start.sh
source /home/nvme01/clash-for-linux/clash.sh && proxy_on
```

然后重启训练：

```bash
cd /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys

GPU_LIST=0,1,2,3,4,5,6,7 \
bash scripts/run_train_trd_v1_dual.sh \
  --project_name intro-example \
  --wandb_entity WorldModel_11
```

重新登录后查看日志：

```bash
tail -n 200 -F /home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_dual.log
```

## 22. 2026-04-18 TRD-v1 H20 `SIGFPE` 专项诊断阶段总结

这一节追加记录 2026-04-18 这一轮围绕 `physical_consistency.cli.train_trd_v1` 的专项排障。注意：本节只总结当前诊断结论，不代表核心问题已经被修复。

### 22.1 这一轮到底在诊断什么

从本轮对话开始到当前节点，主线一直是同一个问题：

- H20 上启动 `TRD-v1 / Stage1 student` 训练；
- forward、loss 计算看起来可以完成；
- 进入 `accelerator.backward(loss_total)` 后，Python 进程被 native 层 `SIGFPE` 杀掉；
- 日志表现为：

```text
Fatal Python error: Floating point exception
...
torch/autograd/graph.py, line 825 in _engine_run_backward
torch/autograd/__init__.py, line 347 in backward
torch/_tensor.py, line 581 in backward
accelerate/accelerator.py, line 2241 in backward
physical_consistency/trainers/trd_v1.py, line 1005 in train
...
traceback : Signal 8 (SIGFPE)
```

因此，本轮不是在调 VideoPhy-2 AutoEval，也不是在调 LingBot generation-only pipeline，而是在调一个新的训练路径：

- Wan2.1 / LingBot student；
- Stage1 checkpoint；
- LoRA adapter 训练；
- memory-efficient Wan block patch；
- gradient checkpointing；
- Flow Matching loss；
- 可选 TRD loss；
- Accelerate / DDP；
- H20 CUDA runtime。

### 22.2 已经解决或排除的旁支问题

#### 22.2.1 启动脚本路径问题

一开始误用了：

```bash
bash scripts/train_trd_v1.sh
```

实际当前项目里的入口是：

```bash
bash scripts/run_train_trd_v1.sh
```

这是启动命令问题，不是训练图本身问题，已经纠正。

#### 22.2.2 DDP ready-twice 报错已经不再是当前阻塞点

之前曾出现过 DDP 报错：

```text
Expected to mark a variable ready only once
...
model.blocks.39.ffn.2.lora_B.weight
```

后来通过以下组合绕开 / 排除：

- `--student_checkpoint_use_reentrant false`
- `--student_lora_merge_mode out_of_place`
- `--student_diagnose_training_graph true`
- `--student_param_grad_trace true`

诊断日志显示：

```text
duplicate_parameter_refs=0
lora_merge_modes=out_of_place:560
```

后续当前阻塞点已经不是 ready-twice，而是 native `SIGFPE`。

#### 22.2.3 OOM 与 SIGFPE 已经区分

在 `checkpoint_mode=none` 的一次 probe 中，如果没有显式指定小分辨率 / 短帧数，会回到默认大尺寸，出现 forward OOM。这个 OOM 是配置问题，不是核心 `SIGFPE`。

正确的小 probe 参数是：

```text
num_frames=17
height=320
width=576
latent_grid=(5,40,72)
seq_len=3600
```

在这个尺寸下：

- `checkpoint_mode=none` 可以跑到 backward 前，但仍然 `SIGFPE`；
- `checkpoint_mode=full` 显存更低，也仍然 `SIGFPE`；
- 最新单卡 `checkpoint_mode=full` 的 backward 前显存只有约 `38.67 GiB allocated`，因此不是 OOM。

#### 22.2.4 TRD loss 已排除

多轮 probe 使用：

```text
--trd_backward_mode off
```

日志显示：

```text
loss_trd=0
loss_total=loss_fm
```

即使完全关闭 TRD loss，仍然在 `loss_fm.backward()` 期间 `SIGFPE`。因此当前问题不是 V-JEPA / VideoMAE teacher loss 的 backward 直接造成。

#### 22.2.5 loss 数值异常已排除

相关日志显示：

```text
loss_total_finite=True
loss_fm_finite=True
sample_sigma_finite=True
sample_timestep_finite=True
```

所以当前不是 Python 层显式 NaN / Inf loss 导致的普通数值错误。

#### 22.2.6 full checkpoint replay 不是必要条件

已测过：

- `student_memory_efficient_checkpoint_mode=full`
- `student_memory_efficient_checkpoint_mode=none`

二者都会在 backward 中触发 `SIGFPE`。因此不能简单归因于 full block checkpoint replay 重算。

#### 22.2.7 `find_unused_parameters` 不是根因

已测过：

- `ddp_find_unused_parameters=True`
- `ddp_find_unused_parameters=False`

二者都会触发 `SIGFPE`。关闭它可以减少 DDP 额外图遍历和 warning，但没有修复核心问题。

#### 22.2.8 多卡 DDP / NCCL 已经基本降级为非主因

最新关键 probe 是单卡：

```text
CUDA_VISIBLE_DEVICES=0
num_gpus=1
world_size=1
student_memory_efficient_checkpoint_mode=full
student_checkpoint_use_reentrant=false
student_lora_merge_mode=out_of_place
trd_backward_mode=off
```

该 run 仍然在 `accelerator.backward()` / `torch.autograd._engine_run_backward` 中触发：

```text
Signal 8 (SIGFPE)
```

这说明问题不是 8 卡 allreduce、NCCL、DDP bucket 或 rank 间同步造成的主问题，而是单卡 backward graph 中的 native kernel / extension / CUDA path。

### 22.3 当前已知显存账本

在 `checkpoint_mode=full`、17 帧、320x576、单卡 probe 中：

```text
after_accelerator_prepare allocated=35.43 GiB
after_teacher_load       allocated=35.43 GiB
before_student_forward   allocated=35.52 GiB
after_student_forward    allocated=38.70 GiB
before_backward          allocated=38.67 GiB
max_allocated            46.32 GiB
```

说明：

- student 常驻模型 / LoRA / DDP wrapper 约 `35 GiB`；
- teacher 采用 `storage_device=cpu`，几乎不增加 GPU 常驻显存；
- full checkpoint 下 forward activation 增量约 `3.2 GiB`；
- 当前单卡 full checkpoint probe 并不接近 96G H20 OOM 边界。

在之前 `checkpoint_mode=none`、同样 17 帧尺寸下：

```text
after_student_forward allocated≈85.28 GiB
before_backward       allocated≈85.25 GiB
```

因此 activation 的显存主因已经清楚：不 checkpoint 时，Wan student forward graph 会额外吃掉约 `50 GiB`。但这只是显存事实，不是当前 `SIGFPE` 的充分解释，因为 full checkpoint 低显存下也会炸。

### 22.4 当前最强结论

当前核心结论是：

> H20 上的 `SIGFPE` 已经收敛到 `Wan2.1 / LingBot student + LoRA + memory-efficient block patch + attention/flash-attn/bf16 CUDA path` 这个训练 backward 图。它不是 TRD teacher loss，不是普通 OOM，不是多卡 DDP/NCCL，同样也不是 `find_unused_parameters` 或 loss NaN。

目前更像是某个 native CUDA backward kernel 在 H20 上被这个训练图触发。候选区域包括：

- Wan attention / flash-attn backward；
- LoRA adapter backward 与大范围 adapter 注入；
- memory-efficient modulation / FFN chunk patch 生成的 backward graph；
- RoPE q/k dtype preserve patch 之后的 dtype 组合；
- H20 + 当前 PyTorch / CUDA / flash-attn 版本的兼容性边界。

### 22.5 为什么 VideoREPA 和 LingBot Stage1 没有遇到这个问题

目前看，最关键的解释是：它们没有覆盖当前这条 backward 图。

#### 22.5.1 VideoREPA 在当前项目中主要是 teacher / eval 角色

VideoREPA 侧主要提供 teacher 特征或评测相关组件。teacher forward / feature extraction 能跑，并不代表 Wan2.1 student 的训练 backward 能跑。

当前 `SIGFPE` 发生在 student 的 `loss_fm.backward()` 中，而不是 teacher forward 中。并且 `trd_backward_mode=off` 时 teacher loss 已经完全关闭，问题仍然存在。

因此，不能把“VideoREPA 没炸”直接等价为“当前 student backward 图也应该没问题”。

#### 22.5.2 LingBot-base / LingBot-Stage1 之前跑通的是 generation / evaluation

此前 H20 上已经跑通的 LingBot-base / LingBot-Stage1 主要是 generation-only 或 evaluation pipeline：

- `no_grad` 推理；
- 没有 optimizer；
- 没有 LoRA adapter backward；
- 没有 full training autograd graph；
- 没有 student block checkpoint backward；
- 没有对 Wan blocks 做训练态大规模梯度回传。

推理路径和训练 backward 路径完全不同。推理稳定只能证明模型权重和基本 forward / sampling 链路可用，不能证明当前 LoRA + patched Wan block 的 backward 链路可用。

#### 22.5.3 当前 TRD-v1 是新的组合系统

当前训练不是原始 LingBot Stage1 训练脚本原封不动复现，也不是 VideoREPA 原始训练脚本原封不动复现，而是一个组合系统：

```text
LingBot / Wan2.1 student
+ Stage1 checkpoint
+ LoRA 560 linear layers
+ memory-efficient Wan block patch
+ optional block checkpointing
+ Flow Matching loss
+ optional TRD teacher loss
+ H20 / flash-attn / Accelerate
```

因此，最合理的判断是：

> 问题不在“VideoREPA 权重坏了”或“LingBot Stage1 权重坏了”，而在我们新搭出来的 TRD-v1 student training path，尤其是 patched Wan student backward graph。

### 22.6 下一步推荐的二分方向

后续如果继续诊断，建议不要再优先纠缠 DDP，而是从训练图本身二分。

#### 22.6.1 缩小 LoRA 范围

例如只训练最后一个 block：

```text
--student_lora_block_start 39
```

如果只训最后 block 可以通过，而全量 560 个 LoRA module 会炸，说明问题和 LoRA 注入范围 / 某些 block 的 adapter backward 有关。

#### 22.6.2 关闭或替换 attention backend

当前脚本会检查 `flash-attn` 可 import，但还没有明确的训练参数用来强制禁用 flash-attn。下一步需要确认 LingBot Wan attention 是否能切到 PyTorch SDPA math backend。

如果关闭 flash-attn 后单卡 backward 通过，那么主因基本就是 attention backward kernel / H20 兼容性。

#### 22.6.3 绕开 memory-efficient Wan block patch

如果原始 Wan block forward + LoRA backward 可以通过，而 memory-efficient patch 路径会 `SIGFPE`，则问题应集中在当前 patch 生成的 autograd graph。

#### 22.6.4 加更细粒度 backward trace

当前 faulthandler 只能看到 Python 主栈在 `_engine_run_backward`，看不到具体哪个 CUDA op。后续可以考虑：

- 对 LoRA A/B 参数记录第一个成功回传的 hook；
- 对 block 级 backward hook 记录最后进入的 block；
- 设置 `CUDA_LAUNCH_BLOCKING=1` 做同步定位；
- 若必要，再用 `TORCH_SHOW_CPP_STACKTRACES=1` 或 CUDA compute sanitizer 做 native 层定位。

### 22.7 本阶段一句话结论

截至 2026-04-18 11:01 左右，核心 `SIGFPE` 尚未修复，但已经从“大概是 DDP / OOM / TRD loss / checkpoint / teacher”的宽泛怀疑，收敛为：

> 单卡也能复现的 Wan student backward native CUDA 问题；最可疑区域是 LoRA + patched Wan block + attention/flash-attn/bf16 在 H20 上的组合。

## 23. 2026-04-18 14:20 后续 probe：已排除项与下一步收敛方向

本节记录 2026-04-18 中午到下午继续做的 TRD-v1 H20 `SIGFPE` probe。这里的“排除”含义是：在当前 17 帧、320x576、low model、单卡最小复现条件下，它不是触发这次 `loss.backward()` 原生 `SIGFPE` 的充分根因；不代表这些路径在完整训练里永远没有显存或性能风险。

### 23.1 新增 probe 时间线

#### 23.1.1 强制绕开 Dao flash-attn backward

代码增加了运行时开关：

```text
PC_FORCE_SDPA_FALLBACK=1
```

它同时 patch：

```text
wan.modules.attention.flash_attention
wan.modules.model.flash_attention
```

日志确认：

```text
PC_FORCE_SDPA_FALLBACK=1: patched wan.modules.attention+wan.modules.model flash_attention -> SDPA fallback (flash-attn backward will NOT be used)
```

结果：仍然在 `before_backward` 之后触发 `Fatal Python error: Floating point exception`。

结论：当前 `SIGFPE` 不是 Dao `flash_attn_varlen` backward 的必要结果。

#### 23.1.2 强制 PyTorch SDPA math backend

继续增加：

```text
PC_FORCE_SDPA_MATH=1
```

日志确认：

```text
PC_FORCE_SDPA_MATH=1: forcing PyTorch SDPA math backend
```

结果：单卡仍然在 backward 中 `SIGFPE`。

结论：当前问题不能再简单归因于 PyTorch fused / memory-efficient SDPA backend；至少在 math backend 下，`SIGFPE` 仍然可复现。

#### 23.1.3 plain python 单进程复现

不再通过 `accelerate launch`，而是直接运行：

```text
python -m physical_consistency.cli.train_trd_v1
```

日志显示：

```text
distributed_type=DistributedType.NO
rank=? local_rank=?
```

结果：仍然在：

```text
torch.autograd.graph._engine_run_backward
accelerate.accelerator.backward
```

内触发 `SIGFPE`。

结论：DDP、NCCL、torchrun / elastic launcher、rank 间同步都不是当前最小复现的主因。

#### 23.1.4 关闭 gradient checkpointing

关键配置：

```text
--gradient_checkpointing false
--student_memory_efficient_modulation false
--student_lora_block_start 39
```

日志确认：

```text
gradient_checkpointing=False
checkpoint_modes=unpatched:40
lora_modules=14
```

结果：仍然在 backward 中 `SIGFPE`。

结论：

- block checkpoint replay 不是当前 `SIGFPE` 的必要条件；
- memory-efficient modulation patch 也不是单独充分根因；
- 只训练最后一个 block 的 14 个 LoRA linear 仍可复现，所以不是“560 个 LoRA 太多”这个规模问题本身。

#### 23.1.5 LoRA fp32 probe

代码增加：

```text
PC_FORCE_LORA_FP32=1
```

日志确认：

```text
lora_dtype=float32
force_lora_fp32=True
```

结果：仍然 `SIGFPE`。

结论：LoRA A/B 参数和 LoRA 分支 matmul 使用 bf16 不是当前 `SIGFPE` 的充分根因。

#### 23.1.6 detach base_out probe

代码增加：

```text
PC_LORA_DETACH_BASE_OUT=1
```

日志确认：

```text
detach_base_out=True
```

结果：仍然 `SIGFPE`。

结论：LoRA wrapper 中 frozen base linear 的输出分支进入 backward 图，不是当前 `SIGFPE` 的充分根因。

#### 23.1.7 detach LoRA input probe

代码增加：

```text
PC_LORA_DETACH_INPUT=1
```

日志确认：

```text
detach_base_out=True
detach_input=True
```

最新一次关键日志为：

```text
Applied standard LoRA to 14 linear layers ...
lora_dtype=float32, force_lora_fp32=True, detach_base_out=True, detach_input=True
distributed_type=DistributedType.NO
gradient_checkpointing=False
checkpoint_modes=unpatched:40
PC_FORCE_SDPA_MATH=1
loss_total_finite=True
before_backward
Fatal Python error: Floating point exception
```

结果：仍然 `SIGFPE`。

结论：LoRA adapter 的输入继续把上游 Wan block / attention / FFN graph 牵进 backward，不是当前 `SIGFPE` 的充分解释。即使把 base 输出和 LoRA 输入都 detach，FM loss 对 `pred` 的 backward 仍会触发 native 崩溃。

### 23.2 到目前为止可以认为“不是当前根因”的项

以下项目已经在当前最小复现里被排除为主因：

- TRD teacher loss / V-JEPA loss：`--trd_backward_mode off` 后仍炸；
- loss 显式 NaN / Inf：`loss_total_finite=True`、`loss_fm_finite=True`；
- OOM：当前 probe backward 前显存约 36-40 GiB allocated，不是 96G H20 OOM；
- 多卡 DDP / NCCL / allreduce：plain python 单进程仍炸；
- `find_unused_parameters`：true / false 均不能解决；
- gradient checkpoint replay：`gradient_checkpointing=False`、`checkpoint_modes=unpatched:40` 仍炸；
- memory-efficient modulation patch：关闭后仍炸；
- Dao flash-attn varlen backward：强制 SDPA fallback 后仍炸；
- PyTorch fused SDPA backend：强制 SDPA math 后仍炸；
- LoRA 数量过大：只保留最后 block 的 14 个 LoRA linear 仍炸；
- LoRA bf16 dtype：强制 LoRA fp32 后仍炸；
- LoRA wrapper 的 frozen base 输出分支：`detach_base_out=True` 后仍炸；
- LoRA 输入向上游传播：`detach_input=True` 后仍炸。

### 23.3 当前剩余最可疑区域

最新证据把问题进一步压缩到：

```text
LoRA adapter output
-> last Wan block / final head / unpatchify / pred path
-> FM loss
-> backward native op
```

也就是说，当前更像是“LoRA 输出被 FM loss 牵到下游 `pred` 路径时，某个 frozen Wan downstream op 的 backward 在 H20 上触发原生 `SIGFPE`”，而不是 LoRA dtype、DDP、checkpoint、TRD 或 flash-attn varlen 本身。

### 23.4 已加入但还需要 H20 运行的下一步：LoRA local loss probe

最新代码已经加入：

```text
PC_LORA_LOCAL_LOSS=1
```

该 probe 在 LoRA forward 内部收集一个 local loss，然后在 `training_step` 中用这个 local loss 替代 FM loss。目的不是训练有效模型，而是二分 backward graph：

```text
LoRA A/B minimal local backward
vs
LoRA output -> pred -> FM loss downstream backward
```

判据：

- 如果 `PC_LORA_LOCAL_LOSS=1` 可以通过 1 个 micro step，则 LoRA A/B 最小 backward 本身可用，问题集中在 LoRA 输出到 `pred/FM loss` 的 downstream Wan 路径；
- 如果 `PC_LORA_LOCAL_LOSS=1` 仍然 `SIGFPE`，则 LoRA A/B 的最小 linear backward 或 H20/PyTorch 原生 linear backward 路径仍然可疑，需要再做纯 LoRA toy backward / `addmm` 级别 probe。

### 23.5 当前一句话结论更新

截至 2026-04-18 14:20，当前 `SIGFPE` 已经不再像是 DDP、TRD、checkpoint、flash-attn、LoRA bf16 或 LoRA 上游传播问题；最强假设已经变成：

> FM loss 从 `pred` 往回拉时，LoRA 输出之后的 Wan downstream backward 路径中存在某个 H20 上不稳定的 native op。下一步用 `PC_LORA_LOCAL_LOSS=1` 把 FM / pred 下游路径切掉，验证 LoRA 最小 backward 是否本身健康。

## 24. 2026-04-18 14:30 LoRA local loss 结果：问题继续缩到 LoRA matmul backward

### 24.1 本轮实验配置

H20 上运行的关键开关：

```text
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
PC_FORCE_LORA_FP32=1
PC_LORA_DETACH_BASE_OUT=1
PC_LORA_DETACH_INPUT=1
PC_LORA_LOCAL_LOSS=1
```

同时：

```text
distributed_type=DistributedType.NO
gradient_checkpointing=False
checkpoint_modes=unpatched:40
lora_modules=14
```

日志确认 local loss probe 生效：

```text
local_loss_probe=True
[PHASE] label=lora_local_loss_probe ... loss_lora_local=0.047227 loss_lora_local_finite=True
[PHASE] label=before_backward ... loss_total=0.047227 loss_total_finite=True
Fatal Python error: Floating point exception
```

### 24.2 新结论

这轮非常关键：`PC_LORA_LOCAL_LOSS=1` 已经不再使用 FM MSE loss，也不再需要从 `pred` 的 downstream path 往回传。loss 直接来自 LoRA forward 内部的：

```text
lora_hidden.float().square().mean() + lora_out.float().mean()
```

并且本轮同时打开了：

```text
detach_base_out=True
detach_input=True
force_lora_fp32=True
```

所以现在可以把上一节的“LoRA output -> pred/FM downstream”假设再往下修正：当前最小触发面已经非常接近 LoRA 分支自身的 CUDA matmul backward。

目前剩余最可疑区域变成：

```text
detached activation
-> LoRA A fp32 linear
-> LoRA B fp32 linear
-> local scalar loss
-> backward native CUDA op
```

这意味着：

- FM loss / `pred` downstream path 不是必要条件；
- frozen Wan base branch 不是必要条件；
- LoRA 输入向更早 Wan block 传播不是必要条件；
- LoRA bf16 不是必要条件；
- 更可疑的是 H20 上某个 `Linear/mm/addmm` backward、TF32/cublasLt 路径、参数 hook 或 Accelerate backward wrapper 与当前环境组合的问题。

### 24.3 已加入的下一步 probe

代码已经继续加入两个更小的开关：

```text
PC_DISABLE_TF32=1
PC_LORA_PARAM_ONLY_LOSS=1
```

其中：

- `PC_DISABLE_TF32=1` 会设置 `torch.backends.cuda.matmul.allow_tf32=False`、`torch.backends.cudnn.allow_tf32=False`，并把 float32 matmul precision 设为 `highest`，用于排查 fp32 LoRA matmul 是否实际走到 TF32 / cublasLt 路径；
- `PC_LORA_PARAM_ONLY_LOSS=1` 会用只依赖 LoRA 参数本身的 tiny loss，不经过任何 LoRA activation matmul，用来判断“只要 backward 到 LoRA 参数就会炸”还是“必须经过 LoRA matmul backward 才会炸”。

下一步优先跑：

```text
PC_DISABLE_TF32=1 + PC_LORA_LOCAL_LOSS=1
```

判据：

- 如果禁用 TF32 后 local loss 通过，说明问题很可能是 H20 + 当前 PyTorch/CUDA 的 TF32/cublasLt matmul backward 路径，直接修复方向就是训练时禁用 TF32；
- 如果禁用 TF32 后 local loss 仍炸，再跑 `PC_LORA_PARAM_ONLY_LOSS=1`；若 param-only 通过，则进一步坐实 LoRA `Linear/mm/addmm` backward 是触发点；若 param-only 也炸，再继续查 hooks / Accelerate backward / optimizer wrapper。

## 25. 2026-04-18 14:38 no-TF32 local loss 结果：TF32 不是充分根因

### 25.1 本轮实验配置和结果

本轮在上一轮 `PC_LORA_LOCAL_LOSS=1` 基础上加入：

```text
PC_DISABLE_TF32=1
```

日志确认：

```text
PC_DISABLE_TF32=1: disabled CUDA TF32 matmul/cudnn paths (matmul.allow_tf32=False, cudnn.allow_tf32=False)
```

同时 local loss 仍然生效：

```text
local_loss_probe=True
[PHASE] label=lora_local_loss_probe ... loss_lora_local=0.0472017 loss_lora_local_finite=True
[PHASE] label=before_backward ... loss_total=0.0472017 loss_total_finite=True
Fatal Python error: Floating point exception
```

### 25.2 新结论

禁用 TF32 后仍然 `SIGFPE`，所以当前不能把问题简单归因为 TF32 matmul 精度路径。剩余嫌疑继续集中在：

```text
LoRA local loss 的 autograd graph
-> LoRA Linear/mm/addmm backward
-> 或 backward hook / Accelerate backward wrapper / CUDA runtime 组合
```

### 25.3 下一步更干净的参数级 probe

代码继续加入：

```text
PC_LORA_PARAM_ONLY_SKIP_FORWARD=1
```

这个开关会在 `training_step` 开头直接返回一个只依赖 LoRA 参数本身的 tiny loss：

```text
LoRA parameter
-> mean / square mean
-> backward
```

它会跳过：

- VAE encode；
- T5 encode；
- Wan student forward；
- LoRA A/B activation matmul；
- FM loss；
- TRD loss。

判据：

- 如果 `PC_LORA_PARAM_ONLY_SKIP_FORWARD=1` 可以通过，则说明只对 LoRA 参数做普通 autograd backward 是健康的，问题需要 LoRA `Linear/mm/addmm` activation backward 才触发；
- 如果它仍然 `SIGFPE`，那就说明问题已经小到“参数本身的 backward / hook / Accelerate backward / 当前 PyTorch 环境”层面，需要继续用 no-hook、plain `loss.backward()`、toy tensor 脚本来切。

## 26. 2026-04-18 14:42 param-only skip-forward 结果：参数 / hook / Accelerate 简单 backward 健康

### 26.1 本轮实验配置和结果

本轮使用：

```text
PC_DISABLE_TF32=1
PC_FORCE_LORA_FP32=1
PC_LORA_PARAM_ONLY_LOSS=1
PC_LORA_PARAM_ONLY_SKIP_FORWARD=1
```

日志确认本轮没有进入 student forward，而是直接使用参数级 tiny loss：

```text
[PHASE] label=lora_param_only_skip_forward_probe ... loss_lora_param=6.31063e-07 loss_lora_param_finite=True
[PHASE] label=before_backward ... loss_total=6.31063e-07 loss_total_finite=True
[PHASE] label=after_backward
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1
```

### 26.2 新结论

这轮通过非常重要，说明以下路径在当前环境里是健康的：

- LoRA 参数本身参与 autograd；
- `lora_b.weight` 参数 grad hook；
- `accelerator.backward()` 的简单参数 loss；
- optimizer step / grad clipping 前后的基本训练框架；
- 不经过 Wan forward 时，当前 35 GiB 常驻模型状态不会自己触发 `SIGFPE`。

因此当前 `SIGFPE` 需要至少经过 LoRA activation matmul 才能触发，最小嫌疑继续缩小为：

```text
activation input
-> lora_A Linear/mm/addmm
-> lora_B Linear/mm/addmm
-> local loss
-> backward
```

### 26.3 下一步：合成 LoRA matmul skip-forward

代码继续加入：

```text
PC_LORA_SYNTHETIC_MATMUL_SKIP_FORWARD=1
```

这个 probe 仍然跳过 VAE/T5/Wan student forward，但会对已加载的 LoRA 模块构造合成随机输入，然后跑：

```text
synthetic detached x
-> lora_A
-> lora_B
-> local scalar loss
-> backward
```

建议第一轮直接覆盖最后 block 的 14 个 LoRA module：

```text
PC_LORA_SYNTHETIC_MODULE_START=0
PC_LORA_SYNTHETIC_MODULE_LIMIT=14
PC_LORA_SYNTHETIC_TOKENS=3600
```

判据：

- 如果合成 LoRA matmul 也 `SIGFPE`，则基本坐实 H20/当前 PyTorch 环境中的 LoRA `Linear/mm/addmm` backward 是触发点，可以继续按 module index 二分；
- 如果合成 LoRA matmul 通过，则纯 LoRA matmul 本身没问题，问题依赖真实 Wan forward 产出的 activation 形状/stride/dtype/数值分布或周边 autograd context。

## 27. 2026-04-18 14:48 synthetic LoRA matmul 结果：纯 LoRA matmul backward 健康

### 27.1 本轮实验配置和结果

本轮使用：

```text
PC_DISABLE_TF32=1
PC_FORCE_LORA_FP32=1
PC_LORA_SYNTHETIC_MATMUL_SKIP_FORWARD=1
PC_LORA_SYNTHETIC_MODULE_START=0
PC_LORA_SYNTHETIC_MODULE_LIMIT=14
PC_LORA_SYNTHETIC_TOKENS=3600
```

日志确认 14 个最后 block LoRA module 全部参与合成 matmul：

```text
PC_LORA_SYNTHETIC_MATMUL_LOSS=1: modules=0:model.blocks.39.self_attn.q ... 13:model.blocks.39.cam_shift_layer ... batch=1 tokens=3600 dtype=float32
[PHASE] label=lora_synthetic_matmul_skip_forward_probe ... loss_lora_synthetic=0.333384 loss_lora_synthetic_finite=True
[PHASE] label=before_backward ... loss_total=0.333384 loss_total_finite=True
[PHASE] label=after_backward
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1
```

### 27.2 新结论

这轮进一步排除了“LoRA Linear/mm/addmm backward 本身在 H20 上必炸”。现在已经确认健康的路径包括：

- LoRA 参数级 backward；
- 14 个 LoRA module 的合成 fp32 `lora_A -> lora_B` matmul backward；
- `lora_b.weight` hook；
- `accelerator.backward()`；
- 禁用 TF32 后的合成 LoRA matmul。

因此，之前真实 forward + local loss 的 `SIGFPE` 更像依赖真实 Wan forward 产出的 activation 条件，例如：

- activation stride / contiguity / storage layout；
- activation 真实数值分布；
- real Wan forward 留下的 autograd context；
- 某个真实 LoRA 输入不是合成 probe 覆盖的标准 contiguous layout。

注意：`_trace_training_phase` 已经在每个 phase 前执行 `torch.cuda.synchronize()`。因此它不像是普通 forward 异步 CUDA error 延迟到 backward 才暴露；更像是 backward 确实在处理真实 LoRA activation 时触发。

### 27.3 下一步：真实 activation 强制 contiguous clone

代码继续加入：

```text
PC_LORA_INPUT_CONTIGUOUS_CLONE=1
PC_LORA_HIDDEN_CONTIGUOUS_CLONE=1
PC_LORA_TRACE_INPUT_META=1
```

下一步回到真实 Wan forward + LoRA local loss，但在 LoRA 分支里强制：

```text
real Wan activation
-> detach
-> fp32
-> contiguous clone
-> lora_A
-> hidden contiguous clone
-> lora_B
-> local loss
```

判据：

- 如果这轮通过，说明真实 activation 的 stride/storage/contiguity 是触发点，后续修复可在 LoRA 分支固定 clone/contiguous；
- 如果仍然 `SIGFPE`，则问题更可能依赖真实 activation 数值分布或真实 forward 周边上下文，需要继续记录每个 LoRA module 的 input meta，并按 module 关闭/替换真实 activation 二分。

## 28. 2026-04-18 14:57 real activation clone 结果：clone 不够，真实 LoRA 计算仍被 autocast 成 bf16

### 28.1 本轮实验配置和结果

本轮回到真实 Wan forward + LoRA local loss，并开启：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DETACH_BASE_OUT=1
PC_LORA_DETACH_INPUT=1
PC_LORA_LOCAL_LOSS=1
PC_LORA_INPUT_CONTIGUOUS_CLONE=1
PC_LORA_HIDDEN_CONTIGUOUS_CLONE=1
PC_LORA_TRACE_INPUT_META=1
```

日志确认 LoRA 分支输入已经是 contiguous：

```text
raw_contig=True
input_dtype=torch.float32
input_contig=True
clone_input=True
clone_hidden=True
```

但关键新发现是：

```text
hidden_dtype=torch.bfloat16
out_dtype=torch.bfloat16
```

也就是说，虽然 LoRA 参数和输入被设为 fp32，真实 Wan forward 外层仍处于：

```text
torch.amp.autocast("cuda", dtype=torch.bfloat16)
```

因此 LoRA `Linear` 实际计算仍被 autocast 到 bf16。synthetic matmul probe 之所以通过，是因为它在 skip-forward 早退分支里，不在这个 autocast context 内。

本轮结果仍然：

```text
[PHASE] label=lora_local_loss_probe ... loss_lora_local=0.0472017 loss_lora_local_finite=True
[PHASE] label=before_backward ... loss_total=0.0472017 loss_total_finite=True
Fatal Python error: Floating point exception
```

### 28.2 新结论

目前最强结论更新为：

```text
真实 Wan forward 的 autocast bf16 LoRA Linear backward
```

才是当前最小触发面。此前 `PC_FORCE_LORA_FP32=1` 只改变了 LoRA 参数 dtype / 输入 cast，但没有阻止外层 autocast 把 matmul 计算降成 bf16。

### 28.3 下一步：在 LoRA 分支内部禁用 autocast

代码继续加入：

```text
PC_LORA_DISABLE_AUTOCAST=1
```

下一轮仍然用真实 Wan forward + local loss，但 LoRA 分支内部会进入：

```text
torch.amp.autocast("cuda", enabled=False)
```

预期日志应从：

```text
hidden_dtype=torch.bfloat16
out_dtype=torch.bfloat16
```

变成：

```text
hidden_dtype=torch.float32
out_dtype=torch.float32
```

判据：

- 如果 `PC_LORA_DISABLE_AUTOCAST=1` 后通过，修复方向就非常明确：LoRA adapter 分支必须在 fp32 下计算，至少 H20 当前环境不能用 bf16 autocast LoRA backward；
- 如果仍然 `SIGFPE`，再按真实 LoRA module 单独开关 / 替换真实 activation 数值继续二分。

## 29. 2026-04-18 15:02 LoRA disable autocast 结果：真实 Wan forward local-loss backward 首次通过

### 29.1 本轮实验配置和结果

本轮在上一轮 real activation clone 的基础上继续加入：

```text
PC_LORA_DISABLE_AUTOCAST=1
```

完整关键开关为：

```text
PC_DISABLE_TF32=1
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
PC_LORA_DETACH_BASE_OUT=1
PC_LORA_DETACH_INPUT=1
PC_LORA_LOCAL_LOSS=1
PC_LORA_INPUT_CONTIGUOUS_CLONE=1
PC_LORA_HIDDEN_CONTIGUOUS_CLONE=1
PC_LORA_TRACE_INPUT_META=1
```

日志确认 LoRA 分支已经不再被外层 bf16 autocast 降精度：

```text
hidden_dtype=torch.float32
out_dtype=torch.float32
disable_autocast=True
```

这次真实 Wan forward + LoRA local loss 成功通过 backward：

```text
[PHASE] label=lora_local_loss_probe ... loss_lora_local=0.047238 loss_lora_local_finite=True
[PHASE] label=before_backward ... loss_total=0.047238 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ...
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1 ... exiting before checkpoint/validation
```

注意最后的退出不是崩溃，而是一步探针的正常停止；本轮没有出现 `Fatal Python error: Floating point exception`。

### 29.2 新结论

到目前为止，已经确认这些路径是好的，不能单独解释 `SIGFPE`：

- 单纯 LoRA 参数参与 backward；
- Accelerate 的 `accelerator.backward()`；
- `lora_b.weight` hook；
- 14 个 LoRA module 的合成 fp32 `lora_A -> lora_B` matmul backward；
- real Wan activation 的 contiguous clone；
- 禁用 TF32；
- 禁用 flash-attn 并强制 SDPA math backend；
- detached base output / detached LoRA input；
- 真实 Wan forward + LoRA local loss，只要 LoRA 分支内部禁用 autocast 并保持 fp32 compute。

因此当前最强定位是：

```text
H20 当前环境下，真实 Wan forward 内部的 LoRA adapter 如果被外层 autocast 成 bf16，
其 backward 会触发 native SIGFPE；LoRA adapter 改为 fp32 compute 后 local-loss backward 通过。
```

这个结果已经从“诊断”进入“修复候选”阶段。下一步不再继续 local loss，而是恢复真实 FM loss，保留 `PC_LORA_DISABLE_AUTOCAST=1`，先仍然保留 detach 开关，验证完整 pred -> FM loss -> backward 是否通过。

### 29.3 下一步：恢复真实 FM loss，保留 LoRA fp32 compute

下一轮去掉：

```text
PC_LORA_LOCAL_LOSS=1
```

保留：

```text
PC_LORA_DISABLE_AUTOCAST=1
PC_FORCE_LORA_FP32=1
PC_LORA_DETACH_BASE_OUT=1
PC_LORA_DETACH_INPUT=1
```

判据：

- 如果通过，说明真实 FM loss 本身没有问题，下一步再去掉 `PC_LORA_DETACH_BASE_OUT=1` / `PC_LORA_DETACH_INPUT=1`，验证完整训练图；
- 如果仍然 `SIGFPE`，说明除了 LoRA bf16 compute 以外，FM loss 牵回来的某段非 LoRA graph 仍有另一个触发点，需要继续缩小。

## 30. 2026-04-18 15:09 真实 FM loss + LoRA fp32 compute + detach 结果：通过

### 30.1 本轮实验配置和结果

本轮去掉了上一轮的 LoRA local loss：

```text
PC_LORA_LOCAL_LOSS=0 / 未设置
```

并保留：

```text
PC_DISABLE_TF32=1
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
PC_LORA_DETACH_BASE_OUT=1
PC_LORA_DETACH_INPUT=1
PC_LORA_INPUT_CONTIGUOUS_CLONE=1
PC_LORA_HIDDEN_CONTIGUOUS_CLONE=1
PC_LORA_TRACE_INPUT_META=1
```

日志确认本轮已经回到真实 FM loss：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320364 loss_fm_finite=True
[PHASE] label=trd_loss_off ... loss_total=0.0320364 loss_total_finite=True
[PHASE] label=before_backward ... loss_total=0.0320364 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ...
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1 ... exiting before checkpoint/validation
```

同时 LoRA trace 继续确认 LoRA 分支是在 fp32 下计算：

```text
hidden_dtype=torch.float32
out_dtype=torch.float32
disable_autocast=True
```

本轮没有出现 `Fatal Python error: Floating point exception`。最后退出仍然是一步探针的预期行为。

### 30.2 新结论

当前已经确认：

- 真实 `pred -> FM loss -> backward` 本身可以通过；
- 只要 LoRA adapter 分支内部禁用 autocast 并保持 fp32 compute，FM loss 不会复现此前 SIGFPE；
- 在 `PC_LORA_DETACH_BASE_OUT=1` 和 `PC_LORA_DETACH_INPUT=1` 同时开启时，完整 student forward、FM loss、Accelerate backward 都是健康的；
- 因此问题进一步集中在：去掉 detach 后，真实训练图中 LoRA 梯度是否能安全穿回 Wan block 的上游 / frozen base 分支。

### 30.3 下一步：先去掉 LoRA input detach，保留 base output detach

下一轮建议只去掉：

```text
PC_LORA_DETACH_INPUT=1
```

但暂时保留：

```text
PC_LORA_DETACH_BASE_OUT=1
PC_LORA_DISABLE_AUTOCAST=1
PC_FORCE_LORA_FP32=1
```

这样可以单独验证：

```text
真实 FM loss -> LoRA adapter fp32 backward -> LoRA input 上游真实 Wan graph
```

是否健康。

判据：

- 如果通过，说明 LoRA 梯度穿回上游 Wan graph 没问题，下一轮再去掉 `PC_LORA_DETACH_BASE_OUT=1`；
- 如果失败，说明 base output branch 还没进入 backward 时，仅 LoRA input 的上游真实 graph 就能触发 SIGFPE，需要继续按 block 内 attention / FFN / cam path 二分。

## 31. 2026-04-18 15:14 W&B 初始化超时：非训练图失败，先禁用 W&B 继续诊断

### 31.1 本轮现象

本轮原计划验证：

```text
去掉 PC_LORA_DETACH_INPUT=1
保留 PC_LORA_DETACH_BASE_OUT=1
保留 PC_LORA_DISABLE_AUTOCAST=1
```

但程序在进入模型加载和训练之前，被 W&B 初始化卡住：

```text
wandb.errors.errors.CommError:
Run initialization has timed out after 90.0 sec.
```

因此这次不是 H20 `SIGFPE`，也不是 backward / CUDA / LoRA 路径问题；它发生在：

```text
runner.initialize_tracking()
accelerator.init_trackers(...)
wandb.init(...)
```

也就是说，本轮没有产生训练图诊断结果。

### 31.2 处理方式

为了继续定位 SIGFPE，代码加入了一个运行时禁用开关：

```text
PC_DISABLE_WANDB=1
```

当该开关开启，或者 `WANDB_MODE` 为 `disabled` / `off` / `none` 时：

- `Accelerator(log_with=None)`，不注册 W&B tracker；
- `initialize_tracking()` 直接跳过 W&B 初始化；
- `log_dict()` 在没有 tracker 时安静跳过日志上报。

这样后续 probe 不再依赖 W&B 网络状态。等训练图稳定后，再恢复 W&B online。

### 31.3 下一步：重复 no-input-detach 实验，但禁用 W&B

下一轮仍然测同一个关键边界：

```text
真实 FM loss -> LoRA adapter fp32 backward -> LoRA input 上游真实 Wan graph
```

只是在环境里额外加：

```text
PC_DISABLE_WANDB=1
WANDB_MODE=disabled
```

判据不变：

- 如果通过，下一步去掉 `PC_LORA_DETACH_BASE_OUT=1`；
- 如果失败，说明仅 LoRA input 上游真实 graph 进入 backward 时仍会触发问题，需要继续按 block 内模块二分。

## 32. 2026-04-18 15:32 no-input-detach 结果：LoRA 梯度穿回上游 Wan graph 通过

### 32.1 本轮实验配置和结果

本轮禁用了 W&B，并重复上一轮因 W&B timeout 未完成的 no-input-detach 实验：

```text
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
PC_LORA_DETACH_BASE_OUT=1
PC_LORA_DETACH_INPUT 未设置
PC_LORA_INPUT_CONTIGUOUS_CLONE=1
PC_LORA_HIDDEN_CONTIGUOUS_CLONE=1
PC_LORA_TRACE_INPUT_META=1
```

日志首先确认 W&B 已被成功跳过：

```text
PC_DISABLE_WANDB/WANDB_MODE disabled: skipping W&B tracker initialization
```

LoRA 应用日志确认本轮确实没有 detach input：

```text
detach_base_out=True
detach_input=False
local_loss_probe=False
disable_autocast=True
```

训练图结果：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320364 loss_fm_finite=True
[PHASE] label=before_backward ... loss_total=0.0320364 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ...
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1 ... exiting before checkpoint/validation
```

本轮没有出现 `Fatal Python error: Floating point exception`。最后退出仍然是一步探针的正常行为。

### 32.2 新结论

现在可以进一步确认：

- W&B 不是训练图问题，禁用后 probe 可以正常继续；
- `PC_LORA_DISABLE_AUTOCAST=1` + `PC_FORCE_LORA_FP32=1` 仍然是关键稳定条件；
- 去掉 `PC_LORA_DETACH_INPUT=1` 后，真实 FM loss 的梯度可以穿过 LoRA adapter，并继续进入 LoRA input 的上游真实 Wan graph；
- 因此 “LoRA input 上游 graph” 不是当前最小触发点。

当前剩下最值得验证的边界是：

```text
PC_LORA_DETACH_BASE_OUT=1
```

也就是 frozen base output 分支是否能完整进入 backward 图。

### 32.3 下一步：去掉 base output detach，验证完整 FM backward 图

下一轮去掉：

```text
PC_LORA_DETACH_BASE_OUT=1
```

同时继续不设置：

```text
PC_LORA_DETACH_INPUT=1
```

保留：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
```

判据：

- 如果通过，说明完整 FM backward 图在 LoRA fp32 compute 下已经可以工作，后续再移除 clone/trace 等诊断开关，进入更接近正式训练的验证；
- 如果失败，则说明 base output 分支进入 backward 后仍有触发点，需要继续按 Wan block 内 self-attn / cross-attn / FFN / cam 分支二分。

## 33. 2026-04-18 15:37 full-graph 结果：完整 FM backward 图通过

### 33.1 本轮实验配置和结果

本轮去掉了最后一个 detach：

```text
PC_LORA_DETACH_BASE_OUT 未设置
PC_LORA_DETACH_INPUT 未设置
```

保留核心修复候选和诊断开关：

```text
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
PC_LORA_INPUT_CONTIGUOUS_CLONE=1
PC_LORA_HIDDEN_CONTIGUOUS_CLONE=1
PC_LORA_TRACE_INPUT_META=1
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
```

LoRA 应用日志确认本轮已经是完整图：

```text
detach_base_out=False
detach_input=False
local_loss_probe=False
disable_autocast=True
```

训练结果：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320364 loss_fm_finite=True
[PHASE] label=before_backward ... loss_total=0.0320364 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ...
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1 ... exiting before checkpoint/validation
```

本轮没有出现 `Fatal Python error: Floating point exception`。最后退出是一步探针的正常行为。

### 33.2 新结论

这是目前最强的修复证据：

```text
完整 FM backward 图 + LoRA fp32 compute + LoRA 内部禁用 autocast = 通过
```

因此可以把此前的问题进一步归纳为：

```text
不是 frozen base 分支本身必炸；
不是 LoRA input 上游真实 Wan graph 必炸；
不是 FM loss 必炸；
不是 SDPA math/fallback 或 flash-attn 单独导致；
而是 LoRA adapter 分支在真实 Wan autocast bf16 环境下参与 backward 时触发 H20 native SIGFPE。
```

当前已经可以把 `PC_FORCE_LORA_FP32=1 + PC_LORA_DISABLE_AUTOCAST=1` 视为修复候选，而不是单纯诊断开关。

### 33.3 下一步：撤掉 clone/trace 诊断开关，只保留核心修复

下一轮去掉：

```text
PC_LORA_INPUT_CONTIGUOUS_CLONE=1
PC_LORA_HIDDEN_CONTIGUOUS_CLONE=1
PC_LORA_TRACE_INPUT_META=1
```

保留：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
```

判据：

- 如果通过，说明 clone/trace 不是必要修复，LoRA fp32 compute 才是核心；
- 如果失败，再把 clone 逐个加回，判断是否除了 autocast 以外还需要 contiguous clone。

## 34. 2026-04-18 15:41 minimal 结果：clone/trace 不是必要修复

### 34.1 本轮实验配置和结果

本轮撤掉了上一轮的 LoRA clone/trace 诊断开关：

```text
PC_LORA_INPUT_CONTIGUOUS_CLONE 未设置
PC_LORA_HIDDEN_CONTIGUOUS_CLONE 未设置
PC_LORA_TRACE_INPUT_META 未设置
```

保留核心修复候选：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

同时仍保留环境隔离开关：

```text
PC_DISABLE_TF32=1
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
```

LoRA 应用日志确认本轮是完整图且没有 clone/trace：

```text
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
```

训练结果：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320364 loss_fm_finite=True
[PHASE] label=before_backward ... loss_total=0.0320364 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ...
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1 ... exiting before checkpoint/validation
```

本轮没有出现 `Fatal Python error: Floating point exception`。

### 34.2 新结论

现在可以排除：

- `PC_LORA_INPUT_CONTIGUOUS_CLONE=1` 是必要修复；
- `PC_LORA_HIDDEN_CONTIGUOUS_CLONE=1` 是必要修复；
- `PC_LORA_TRACE_INPUT_META=1` 对行为有保护作用。

也就是说，clone/trace 只是诊断辅助，不是修复关键。当前最小有效修复候选进一步收敛为：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

### 34.3 下一步：去掉 PC_DISABLE_TF32

下一轮建议去掉：

```text
PC_DISABLE_TF32=1
```

保留：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
```

判据：

- 如果通过，说明禁用 TF32 也不是必要条件；
- 如果失败，则说明 LoRA fp32 compute 在 H20 上还需要避开 TF32 matmul 路径，后续正式修复需保留 `PC_DISABLE_TF32=1` 或等效 matmul precision 设置。

## 35. 2026-04-18 15:45 TF32 default 结果：禁用 TF32 不是必要条件

### 35.1 本轮实验配置和结果

本轮去掉了：

```text
PC_DISABLE_TF32=1
```

保留核心修复候选：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

同时仍保留 attention 隔离开关：

```text
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
```

LoRA 应用日志确认仍是最小 LoRA 修复配置：

```text
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
```

训练结果：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320678 loss_fm_finite=True
[PHASE] label=before_backward ... loss_total=0.0320678 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ...
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1 ... exiting before checkpoint/validation
```

本轮没有出现 `Fatal Python error: Floating point exception`。

### 35.2 新结论

可以排除：

```text
PC_DISABLE_TF32=1
```

是必要修复。TF32 默认状态下，只要 LoRA adapter 分支保持 fp32 compute 并禁用 autocast，完整 FM backward 图仍然通过。

当前最小有效修复候选仍然是：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

还没有排除的隔离项是：

```text
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
```

### 35.3 下一步：去掉 PC_FORCE_SDPA_MATH，保留 SDPA fallback

下一轮建议去掉：

```text
PC_FORCE_SDPA_MATH=1
```

保留：

```text
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

这样仍然不使用 Wan/flash-attn 的原始 `flash_attention`，但允许 PyTorch SDPA 自己选择 backend。

判据：

- 如果通过，说明强制 SDPA math backend 不是必要条件；
- 如果失败，则说明 PyTorch SDPA 非 math backend 仍可能触发问题，后续需要保留 `PC_FORCE_SDPA_MATH=1`。

## 36. 2026-04-18 15:50 SDPA default 结果：强制 SDPA math 不是必要条件

### 36.1 本轮实验配置和结果

本轮去掉了：

```text
PC_FORCE_SDPA_MATH=1
```

保留：

```text
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

日志确认仍然 patch 到 SDPA fallback，但没有强制 SDPA math backend：

```text
PC_FORCE_SDPA_FALLBACK=1: patched wan.modules.attention+wan.modules.model flash_attention -> SDPA fallback
```

本轮没有出现：

```text
PC_FORCE_SDPA_MATH=1: forcing PyTorch SDPA math backend
```

LoRA 应用日志确认仍是最小 LoRA 修复配置：

```text
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
```

训练结果：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320519 loss_fm_finite=True
[PHASE] label=before_backward ... loss_total=0.0320519 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ...
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1 ... exiting before checkpoint/validation
```

本轮没有出现 `Fatal Python error: Floating point exception`。

### 36.2 新结论

可以排除：

```text
PC_FORCE_SDPA_MATH=1
```

是必要修复。也就是说，在 `PC_FORCE_SDPA_FALLBACK=1` 仍然生效的情况下，PyTorch SDPA 默认 backend 可以通过。

目前还没有排除的 attention 隔离项只剩：

```text
PC_FORCE_SDPA_FALLBACK=1
```

当前最小修复候选仍然是：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

### 36.3 下一步：去掉 SDPA fallback，回到默认 Wan attention

下一轮去掉：

```text
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
```

保留：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

判据：

- 如果通过，说明 attention fallback / SDPA patch 全部不是必要修复，核心就是 LoRA fp32 compute；
- 如果失败，说明默认 Wan attention / flash-attn 路径仍有问题，正式训练需保留 `PC_FORCE_SDPA_FALLBACK=1`。

## 37. 2026-04-18 15:54 default attention 结果：attention fallback 不是必要修复

### 37.1 本轮实验配置和结果

本轮去掉了全部 attention 隔离开关：

```text
PC_FORCE_SDPA_FALLBACK 未设置
PC_FORCE_SDPA_MATH 未设置
```

保留当前最小 LoRA 修复候选：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

日志中没有出现 SDPA fallback patch：

```text
PC_FORCE_SDPA_FALLBACK=1: patched ...
```

也没有出现强制 SDPA math：

```text
PC_FORCE_SDPA_MATH=1: forcing PyTorch SDPA math backend
```

LoRA 应用日志确认仍是最小 LoRA 修复配置：

```text
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
```

训练结果：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320554 loss_fm_finite=True
[PHASE] label=before_backward ... loss_total=0.0320554 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ...
[MEM PROBE] reached max_train_micro_steps=1 global_step=1 micro_step=1 ... exiting before checkpoint/validation
```

本轮没有出现 `Fatal Python error: Floating point exception`。

### 37.2 新结论

可以排除：

```text
PC_FORCE_SDPA_FALLBACK=1
PC_FORCE_SDPA_MATH=1
```

是必要修复。默认 Wan attention / flash-attn 路径在 LoRA fp32 compute + LoRA 内部禁用 autocast 时可以通过。

当前最小有效修复候选进一步收敛为：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

至此已经排除：

- TRD/V-JEPA loss；
- W&B；
- checkpointing；
- memory-efficient modulation；
- LoRA clone/trace；
- TF32；
- SDPA math；
- SDPA fallback / attention patch。

### 37.3 下一步：去掉 PC_FORCE_LORA_FP32，仅保留 LoRA disable autocast

下一轮去掉：

```text
PC_FORCE_LORA_FP32=1
```

保留：

```text
PC_LORA_DISABLE_AUTOCAST=1
```

判据：

- 如果通过，说明仅 LoRA 分支内部禁用 autocast 就足够；
- 如果失败，说明 LoRA 参数 / LoRA input / LoRA matmul 必须显式保持 fp32，正式修复需要同时保留 fp32 LoRA dtype 和 disable autocast。

## 38. 2026-04-18 15:58 no-force-fp32 结果：失败，LoRA fp32 dtype 是必要条件

### 38.1 本轮配置

本轮去掉：

```text
PC_FORCE_LORA_FP32=1
```

保留：

```text
PC_LORA_DISABLE_AUTOCAST=1
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
REQUIRE_TRAIN_FLASH_ATTN=0
```

同时使用默认 attention 路径，没有启用 SDPA fallback/math 探针，也没有启用 clone/trace/local-loss 诊断开关。

LoRA 应用日志显示：

```text
lora_dtype=bfloat16
force_lora_fp32=False
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
```

### 38.2 结果

forward 与 loss 仍然正常：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320554 loss_fm_finite=True
[PHASE] label=before_backward ... loss_total=0.0320554 loss_total_finite=True
```

但进入 backward 后再次触发原始问题：

```text
Fatal Python error: Floating point exception
...
torch.autograd.graph.py line 825 in _engine_run_backward
accelerate/accelerator.py line 2241 in backward
physical_consistency/trainers/trd_v1.py line 1050 in train
```

### 38.3 结论

这轮是关键反证：仅仅在 LoRA 分支内部禁用 autocast 不够。

当 LoRA 参数仍是 bfloat16 时，即使 `PC_LORA_DISABLE_AUTOCAST=1` 让内部 autocast 关闭，H20 backward 仍会在 native autograd/CUDA 扩展层触发 SIGFPE。

因此当前最小有效修复不只是禁用 LoRA autocast，而是必须同时满足：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

这说明根因高度集中在 H20 上 bf16 LoRA adapter backward 路径，而不是 attention backend、TF32、TRD loss、W&B、checkpointing、memory-efficient modulation、clone/stride 或 loss 数值本身。

### 38.4 附：本轮 log 未覆盖原因

本轮命令末尾显示为：

```text
2>&1 | tee logs/train_trd_v1_low.logh true \se \ull \
```

这会把输出写到 `logs/train_trd_v1_low.logh` 等错误目标，而不是 `logs/train_trd_v1_low.log`。因此 IDE 中的 `train_trd_v1_low.log` 仍可能停留在上一轮通过的结果。

下一轮命令需要确保使用：

```text
2>&1 | tee logs/train_trd_v1_low.log
```

### 38.5 下一步：用最小有效修复跑 3 个 micro step

下一轮恢复：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

并把 `--max_train_micro_steps` 从 1 提到 3，用默认 attention 路径确认不是单步偶然通过。

## 39. 2026-04-18 16:06 最小有效修复 3-step 结果：通过

### 39.1 本轮配置

本轮恢复最小有效修复候选：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

同时保持：

```text
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
REQUIRE_TRAIN_FLASH_ATTN=0
--trd_backward_mode off
--max_train_micro_steps 3
```

没有启用 `PC_FORCE_SDPA_FALLBACK` / `PC_FORCE_SDPA_MATH`，因此走默认 attention 路径。也没有启用 clone/trace/local-loss 等额外诊断开关。

LoRA 应用日志确认：

```text
lora_dtype=float32
force_lora_fp32=True
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
```

### 39.2 结果

第 1 个 micro step 完整通过 forward / loss / backward：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_fm_loss ... loss_fm=0.0320554 loss_fm_finite=True
[PHASE] label=before_backward ... loss_total=0.0320554 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ... loss_total=0.0321 ... peak_mem=46.03GiB
```

第 2、3 个 micro step 也继续通过：

```text
[PROGRESS] epoch=1/5 global_step=2/8350 micro_step=2 ... loss_total=0.0408 ... peak_mem=38.48GiB
[PROGRESS] epoch=1/5 global_step=3/8350 micro_step=3 ... loss_total=0.0826 ... peak_mem=46.15GiB
[MEM PROBE] reached max_train_micro_steps=3 global_step=3 micro_step=3 peak_mem=46.15GiB; exiting before checkpoint/validation
```

本轮没有出现 `Fatal Python error: Floating point exception`。

### 39.3 结论

这轮把 `--max_train_micro_steps` 从 1 提到 3 后仍然稳定通过，说明之前的通过不是单步偶然现象。

结合第 38 轮去掉 `PC_FORCE_LORA_FP32=1` 后立刻失败，可以将当前最小有效修复确定为：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

当前最合理根因表述：

> H20 上 bf16 LoRA adapter backward 路径会在 native autograd/CUDA 扩展层触发 SIGFPE；把 LoRA 参数与 LoRA 分支 matmul 固定为 fp32 可规避该问题。

### 39.4 关于本轮 log 文件没有出现在 IDE 的原因

H20 命令中的：

```text
2>&1 | tee logs/train_trd_v1_low.log
```

只会写入 H20 当前仓库：

```text
/home/nvme03/workspace/world_model_phys/PHYS/world_model_phys/logs/train_trd_v1_low.log
```

IDE 当前打开的是本地/另一份镜像：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/train_trd_v1_low.log
```

这两个路径不是同一个文件视图，因此 H20 上的 `tee` 不会自动刷新 IDE 里打开的 `/home/hj/.../train_trd_v1_low.log`。需要以 H20 stdout 为准，或单独把 H20 的 `logs/train_trd_v1_low.log` 同步/复制回 `/home/hj/.../train_trd_v1_low.log`。

### 39.5 下一步

下一轮建议在最小修复固定后，先重新打开完整 TRD backward，但仍保留 W&B disabled，观察是否还有新的 TRD/teacher 相关问题：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
--trd_backward_mode full
--max_train_micro_steps 3
```

## 40. 2026-04-18 16:15 TRD full 3-step 结果：通过

### 40.1 本轮配置

本轮在第 39 轮最小有效修复基础上，重新打开完整 TRD backward：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
--trd_backward_mode full
--max_train_micro_steps 3
```

同时保持 W&B disabled：

```text
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
```

LoRA 应用日志继续确认：

```text
lora_dtype=float32
force_lora_fp32=True
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
```

### 40.2 关键观测

本轮已经进入 teacher encode：

```text
[PHASE] label=before_teacher_encode ... video=shape(3, 17, 320, 576) dtype=torch.float32
[PHASE] label=after_teacher_encode ... teacher_tokens=shape(1, 32, 576, 768) dtype=torch.float32 requires_grad=False
```

student forward 正常：

```text
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
student_tokens=shape(1, 5, 720, 768) dtype=torch.bfloat16 ... requires_grad=True
```

TRD loss 正常且 finite：

```text
[PHASE] label=after_trd_loss ... loss_total=0.0335121 loss_total_finite=True
loss_trd=0.0145668 loss_trd_finite=True
loss_trd_spatial=0.00178301 loss_trd_spatial_finite=True
loss_trd_temporal=0.0127838 loss_trd_temporal_finite=True
```

完整 backward 通过：

```text
[PHASE] label=before_backward ... loss_total=0.0335121 loss_total_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ... loss_total=0.0335 loss_fm=0.0321 loss_trd=0.0146
```

第 2、3 个 micro step 也继续通过：

```text
[PROGRESS] epoch=1/5 global_step=2/8350 micro_step=2 ... loss_total=0.0437 loss_fm=0.0408 loss_trd=0.0296
[PROGRESS] epoch=1/5 global_step=3/8350 micro_step=3 ... loss_total=0.0843 loss_fm=0.0826 loss_trd=0.0177
[MEM PROBE] reached max_train_micro_steps=3 global_step=3 micro_step=3 peak_mem=46.18GiB; exiting before checkpoint/validation
```

本轮没有出现 `Fatal Python error: Floating point exception`。

### 40.3 结论

这是一个明显进展：打开完整 TRD backward 后仍能稳定通过 3 个 micro step。

因此可以进一步排除：

- V-JEPA teacher encode；
- TRD spatial loss；
- TRD temporal loss；
- TRD loss 参与 student token backward；
- teacher tokens dtype/shape 路径；
- TRD full backward 本身。

当前根因判断进一步收敛为：

```text
H20 + bf16 LoRA adapter backward
```

当前最小有效修复仍是：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

### 40.4 下一步

建议继续保持 W&B disabled，把 `--max_train_micro_steps` 从 3 提到 20，观察是否存在较晚出现的 batch/数据相关问题。

如果 20 step 通过，再考虑两件事：

1. 去掉 `--max_train_micro_steps` 开始真实训练；
2. 或先把 W&B 从 disabled 改回 online，确认只是网络/初始化偶发问题，不影响训练本体。

## 41. 2026-04-18 16:22 TRD full 20-step 结果：通过，原 SIGFPE 问题已被规避

### 41.1 本轮配置

本轮在完整 TRD backward 基础上，把探针步数从 3 提到 20：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
REQUIRE_TRAIN_FLASH_ATTN=0
```

LoRA 应用日志确认仍是最小有效修复：

```text
lora_dtype=float32
force_lora_fp32=True
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
```

### 41.2 结果

第 1 step 完整通过 teacher encode、student forward、FM loss、TRD loss 和 backward：

```text
[PHASE] label=after_teacher_encode ... teacher_tokens=shape(1, 32, 576, 768) dtype=torch.float32 requires_grad=False
[PHASE] label=after_student_forward ... pred=shape(16, 5, 40, 72) dtype=torch.float32 ... requires_grad=True
[PHASE] label=after_trd_loss ... loss_total=0.0335121 loss_total_finite=True loss_trd=0.0145668 loss_trd_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ... loss_total=0.0335 loss_fm=0.0321 loss_trd=0.0146
```

随后连续跑到第 20 step：

```text
[PROGRESS] epoch=1/5 global_step=10/8350 micro_step=10 ... loss_total=0.0846 loss_fm=0.0835 loss_trd=0.0114
[PROGRESS] epoch=1/5 global_step=15/8350 micro_step=15 ... loss_total=0.0492 loss_fm=0.0491 loss_trd=0.0009
[PROGRESS] epoch=1/5 global_step=20/8350 micro_step=20 ... loss_total=0.0634 loss_fm=0.0629 loss_trd=0.0052
[MEM PROBE] reached max_train_micro_steps=20 global_step=20 micro_step=20 peak_mem=38.56GiB; exiting before checkpoint/validation
```

本轮没有出现：

```text
Fatal Python error: Floating point exception
```

也没有出现 loss NaN/Inf 或 CUDA OOM。

### 41.3 结论

可以认为原始 H20 SIGFPE 问题已经被当前修复规避。

精确表述是：

```text
H20 上 bf16 LoRA adapter backward 会触发 native SIGFPE；
将 LoRA 参数/输入/LoRA 分支 matmul 固定为 fp32，并在 LoRA 分支内部禁用 autocast 后，
full TRD backward 已稳定通过 20 个 micro step。
```

当前应保留的最小修复：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

已经排除为主因的路径包括：

- W&B；
- TF32；
- SDPA fallback/math/default attention；
- V-JEPA teacher encode；
- TRD spatial/temporal loss；
- TRD full backward；
- checkpointing；
- memory-efficient modulation；
- LoRA clone/stride；
- loss 数值异常。

### 41.4 尚未验证

本轮仍然是 probe 运行，并在第 20 step 手动退出：

```text
max_train_micro_steps=20
```

因此还没有验证：

- 无 `--max_train_micro_steps` 的长训练；
- checkpoint 保存；
- validation；
- W&B online 初始化与同步。

### 41.5 下一步

下一步建议去掉 `--max_train_micro_steps`，保持 W&B disabled，先跑一次真实训练流程。这样可以验证 checkpoint/validation 前后的路径。

如果真实训练也稳定，再单独恢复 W&B online。

## 42. 2026-04-18 16:32 real training 运行中：已超过 35 step，训练步稳定

### 42.1 本轮配置

本轮去掉了 `--max_train_micro_steps`，开始真实训练流程。关键修复仍然保留：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

W&B 仍然关闭：

```text
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
```

本轮命令指定了低分辨率/短帧数：

```text
--num_frames 17
--height 320
--width 576
```

log 中序列几何确认：

```text
[SEQ GEOM] num_frames=17 latent_grid=(5,40,72) patch_size=(1, 2, 2) seq_len=3600
```

teacher/student 输入也确认当前视频张量是：

```text
video=shape(3, 17, 320, 576)
```

### 42.2 结果

LoRA 配置继续正确：

```text
lora_dtype=float32
force_lora_fp32=True
detach_base_out=False
detach_input=False
local_loss_probe=False
clone_input=False
clone_hidden=False
trace_input_meta=False
disable_autocast=True
trainable=2572288/18548215872
```

训练计划：

```text
epochs=5
micro_steps_per_epoch=1670
optimizer_steps_per_epoch=1670
total_optimizer_steps=8350
grad_accum=1
dataset_samples=1670
world_size=1
```

真实训练已经超过 20-step probe，并继续到至少 step 35：

```text
[PROGRESS] epoch=1/5 global_step=20/8350 micro_step=20 ... loss_total=0.0634 loss_fm=0.0629 loss_trd=0.0052
[PROGRESS] epoch=1/5 global_step=25/8350 micro_step=25 ... loss_total=0.0337 loss_fm=0.0334 loss_trd=0.0024
[PROGRESS] epoch=1/5 global_step=30/8350 micro_step=30 ... loss_total=0.2223 loss_fm=0.2214 loss_trd=0.0090
[PROGRESS] epoch=1/5 global_step=35/8350 micro_step=35 ... loss_total=0.0611 loss_fm=0.0610 loss_trd=0.0004
```

截至 step 35：

- 未出现 `Fatal Python error: Floating point exception`；
- 未出现 CUDA OOM；
- 未出现 loss NaN/Inf；
- peak memory 主要在约 `38.56GiB` 与 `46.18GiB` 两档之间波动；
- optimizer step 在持续推进，`global_step` 正常增长。

### 42.3 当前判断

对于当前实验规格：

```text
low model
17 frames
320 x 576
1 GPU
TRD full backward
LoRA block_start=39
```

原先遇到的训练步 SIGFPE/OOM 问题已经解决或规避。

LoRA 训练路径也已经恢复到可训练状态：

```text
trainable_tensors=28
trainable_params=2572288
after_backward 正常
optimizer step 正常推进
```

这说明已经不再是“LoRA 完全没有进入训练”的状态。

### 42.4 仍需最后确认的边界

还不能说整个训练任务 100% 完成验证，因为目前尚未观察到：

- 第一次 checkpoint 保存成功；
- validation 成功；
- W&B online 恢复后成功初始化与同步；
- 更高规格配置，例如默认 `81 frames / 480 x 832`。

### 42.5 关于 log 文件名

本轮命令末尾仍然出现了多余尾巴：

```text
2>&1 | tee logs/train_trd_v1_low_real.loge \se \ull \
```

这会写到 `logs/train_trd_v1_low_real.loge`，不是预期的：

```text
logs/train_trd_v1_low_real.log
```

终端输出本身是有效训练结果；后续建议复制命令时删除 `\se \ull \` 尾巴，并固定使用干净的 `tee logs/train_trd_v1_low_real.log`。

## 43. 2026-04-18 16:46 81f / 288x496 运行中：显存稳定，memory-efficient 生效

### 43.1 本轮配置

本轮把帧数提升到 81，并按接近原始比例降低空间分辨率：

```text
--num_frames 81
--height 288
--width 496
```

关键修复仍然保留：

```text
PC_FORCE_LORA_FP32=1
PC_LORA_DISABLE_AUTOCAST=1
```

本轮打开显存节省路径：

```text
--student_memory_efficient_modulation true
--gradient_checkpointing true
--student_checkpoint_use_reentrant false
--student_memory_efficient_checkpoint_mode full
```

### 43.2 关键日志

memory-efficient modulation 生效：

```text
Memory-efficient modulation patched 40 blocks for low_noise_model (ffn_chunk_size=512)
```

gradient checkpointing 生效：

```text
Gradient checkpointing wrapped 39 memory-efficient Wan blocks in low_noise_model (use_reentrant=False; full block replay)
Gradient checkpointing skipped 1 blocks for low_noise_model (indices=20)
```

LoRA 配置继续正确：

```text
lora_dtype=float32
force_lora_fp32=True
disable_autocast=True
trainable=2572288/18548215872
```

序列几何确认：

```text
[SEQ GEOM] num_frames=81 latent_grid=(21,36,62) patch_size=(1, 2, 2) seq_len=11718
video=shape(3, 81, 288, 496)
student_tokens=shape(1, 21, 558, 768)
```

第 1 step 完整通过：

```text
[PHASE] label=after_trd_loss ... loss_total=0.0444592 loss_total_finite=True loss_trd=0.0104975 loss_trd_finite=True
[PHASE] label=after_backward ...
[PROGRESS] epoch=1/5 global_step=1/8350 micro_step=1 ... loss_total=0.0445 loss_fm=0.0434 loss_trd=0.0105 ... peak_mem=46.12GiB
```

随后继续稳定推进：

```text
[PROGRESS] global_step=2 ... peak_mem=45.05GiB
[PROGRESS] global_step=3 ... peak_mem=46.32GiB
[PROGRESS] global_step=4 ... peak_mem=46.32GiB
[PROGRESS] global_step=5 ... peak_mem=46.32GiB
[PROGRESS] global_step=6 ... peak_mem=46.32GiB
[PROGRESS] global_step=7 ... peak_mem=45.04GiB
```

### 43.3 结论

`81 frames / 288 x 496` 当前看起来稳定。显存没有明显上升不是坏信号，主要原因是：

- memory-efficient modulation 已经 patch 了 40 个 block；
- gradient checkpointing 已经包住 39 个 block；
- H20 SIGFPE 修复仍然开启；
- 当前 LoRA 只训练 block 39 的 14 个模块，trainable params 约 2.57M；
- checkpointing 用更多计算换更低显存，所以 step time 上升到约 21-31 秒是正常的。

因此不能把“显存没有很高”解读成 LoRA 没有梯度。当前日志至少说明：

```text
trainable_tensors=28
trainable_params=2572288
loss_total_finite=True
after_backward 正常
global_step 正常增长
optimizer step 正常推进
```

### 43.4 仍需增强的 LoRA 梯度可见性

当前 console progress 没有打印 `grad_norm`，所以肉眼不能直接看到 LoRA 梯度范数。实际上训练循环已经计算：

```text
grad_norm = clip_grad_norm_(_all_trainable_params(), max_grad_norm)
```

而当前 `_all_trainable_params()` 基本就是 LoRA trainable params，因此该 `grad_norm` 就是 LoRA 梯度是否正常回传的直接信号。

后续代码已补充：在 `[PROGRESS]` 行中打印 `grad_norm=...`。下一次 pull 后的新 run 可直接通过 progress 行确认 LoRA 梯度是否 finite/非零。

## 44. 2026-04-18 正式训练策略：4 卡、W&B disabled、本地 JSONL 指标

### 44.1 81f / 288x496 probe 结果

本轮 `81 frames / 288 x 496` 已跑完 20-step probe：

```text
[PROGRESS] epoch=1/5 global_step=20/8350 micro_step=20 ... loss_total=0.0599 loss_fm=0.0595 loss_trd=0.0039 ... peak_mem=45.05GiB
[MEM PROBE] reached max_train_micro_steps=20 global_step=20 micro_step=20 peak_mem=45.05GiB; exiting before checkpoint/validation
```

结论：

- 81 帧路径稳定；
- memory-efficient modulation 与 gradient checkpointing 生效；
- H20 SIGFPE 修复仍有效；
- 单卡显存只有约 45-46GiB，仍有明显余量。

### 44.2 下一步正式训练目标

用户当前可用 GPU：

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
```

要真正使用 4 张卡，不能只运行：

```text
python -m physical_consistency.cli.train_trd_v1 --num_gpus 4
```

而需要用：

```text
python -m torch.distributed.run --standalone --nproc_per_node=4 -m physical_consistency.cli.train_trd_v1 ...
```

这样 Accelerate 才会看到 `WORLD_SIZE=4`，训练才是 4 进程 DDP。

### 44.3 分辨率选择

由于 81f / 288x496 只有约 45-46GiB 峰值，下一轮正式训练可以回到默认空间分辨率：

```text
81 frames / 480 x 832
```

这会显著提高 token 数：

```text
288x496: latent_grid=(21,36,62), seq_len=11718
480x832: latent_grid=(21,60,104), seq_len=32760
```

预计显存会明显上升，更接近 H20 的合理利用区间。若 480x832 OOM，回退优先级为：

```text
448 x 768
416 x 720
384 x 672
```

### 44.4 W&B 替代记录

W&B 当前不稳定，因此保持：

```text
WANDB_MODE=disabled
PC_DISABLE_WANDB=1
```

为了替代 W&B，本地新增：

```text
PC_LOCAL_METRICS_PATH=logs/<run_name>_metrics.jsonl
```

后续所有 `log_dict()` payload 会追加写入 JSONL，包括 train loss、lr、grad_norm、epoch summary、validation/checkpoint 相关指标。这样即使 W&B 不可用，也可以从本地文件追溯训练过程。

### 44.5 LoRA 梯度确认

代码已把 `[PROGRESS]` 行增强为打印：

```text
grad_norm=...
```

当前只有 LoRA 参数可训练，因此该 `grad_norm` 基本就是 LoRA trainable params 的梯度范数。正式训练时判据：

```text
grad_norm finite 且不是长期 0
```

即可确认 LoRA 梯度仍在正常回传。

### 44.6 速度与溯源取舍

为了最快训练：

- W&B disabled；
- training 主流程不跑外部 validation；
- 每个 epoch 保存 checkpoint；
- 所有 train metrics 写入本地 JSONL；
- 训练完成后再单独跑评估。

建议正式训练参数：

```text
--validation_every_epochs 0
--validation_every_steps 0
--save_every_n_epochs 1
```

这比每个 epoch 内暂停训练做外部 validation 更快，同时每个 epoch checkpoint 都可用于回溯。

## 45. 4-GPU 正式训练首启失败：save_every_n_epochs CLI 未暴露

### 45.1 现象

4 卡启动命令使用：

```text
python -m torch.distributed.run --standalone --nproc_per_node=4 ...
```

训练进程尚未进入模型加载或前后向阶段，四个 rank 都在参数解析阶段失败：

```text
train_trd_v1.py: error: unrecognized arguments: --save_every_n_epochs 1
```

因此这次失败不是 OOM，不是 H20 SIGFPE，也不是 LoRA 梯度再次断开，而是 CLI parser 没有暴露已有配置项。

### 45.2 根因

配置文件和训练循环已经支持：

```text
save_every_n_epochs
```

但 `parse_args()` 没有注册：

```text
--save_every_n_epochs
```

导致命令行覆盖每 epoch checkpoint 策略时被 argparse 拒绝。

### 45.3 修复

已补齐：

```text
parser.add_argument("--save_every_n_epochs", type=int, default=None)
```

并在配置归一化阶段按非负整数处理：

```text
payload.setdefault("save_every_n_epochs", 0)
save_every_n_epochs >= 0
```

修复后可以继续使用正式训练命令中的：

```text
--save_every_n_epochs 1
```

用于每个 epoch 保存 checkpoint，同时关闭 validation 以保持训练主流程最快。

## 46. Validation 改为非致命：保证 5 个 epoch 权重优先

### 46.1 需求

正式训练会持续很久，核心优先级是：

```text
必须完成 5 个 epoch，并产出 5 个 epoch checkpoint。
```

每个 epoch 结束后仍然希望尝试 validation：

- 先用当前 epoch 权重生成 mini-val 视频；
- mini-val manifest 当前为 8 个样本；
- 5 个 epoch 共期望保存 40 个 validation 视频；
- 然后尝试 VideoPhy-2 打分。

但 validation 不是训练的生死开关：

```text
即使 validation 生成失败、VideoPhy-2 环境失败、summary 缺失，也不能中断训练。
```

### 46.2 当前代码路径

`validation_runtime_mode=pause_external` 的设计是：

1. 保存 `_candidate_epoch_N` checkpoint；
2. 释放训练 runtime 和 CUDA cache；
3. 用释放后的 GPU 显存跑 validation generation；
4. 调用 VideoPhy-2 脚本，脚本会通过 `scripts/lib_videophy2_env.sh` 解析 `phys-videophy` 环境；
5. 恢复训练 runtime；
6. 根据 validation 结果决定是否更新 best checkpoint。

这意味着 validation 期间训练会暂停，显存会被释放给生成/评测使用，然后再恢复训练。

### 46.3 修复

新增：

```text
validation_fail_fast: false
--validation_fail_fast false
```

当外部 validation 失败时：

- 写入 `_candidate_epoch_N/validation_error.txt`，包含 traceback；
- 主训练 log 打印 `[VALIDATION FAILED]` 和 `[VALIDATION NONFATAL]`；
- 本地 metrics 写入 `val/failed=1`；
- 清理临时 candidate checkpoint；
- 恢复训练并继续下一个 epoch。

只有显式设置：

```text
--validation_fail_fast true
```

才恢复旧行为：validation 一失败就终止训练。

### 46.4 Checkpoint 优先级

正式训练命令必须保留：

```text
--save_every_n_epochs 1
--validation_every_epochs 1
--validation_fail_fast false
```

训练循环的顺序是先保存正式 epoch checkpoint，再进入 validation。因此即使 validation 失败，`epoch_1`、`epoch_2`、... 这类权重也已经落盘。
