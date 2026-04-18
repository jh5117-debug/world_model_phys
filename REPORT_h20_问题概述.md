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
