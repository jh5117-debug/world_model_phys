# Physical_Consistency 真实仓库审计与综合 Review v1

## 0. 审计范围与结论

这份文档基于三部分真实代码做交叉审计：

1. 你当前子项目：
   - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency`
2. 真实 `VideoREPA` 仓库：
   - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/VideoREPA`
3. 真实 `videophy` 仓库：
   - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/videophy`
4. 本地 clone 的真实 LingBot / Stage 1 代码：
   - `/home/hj/Multi-View-Physically-Consistent-World-Model/code/lingbot-world`
   - `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune`

审计时间：

- `2026-04-06`

真实 repo 提交：

- `VideoREPA`: `8d581dc301e18546a1808436c724f59d259d09f9`
- `videophy`: `01d366c979e4a82875852adbf427ccc036e7373a`

文件规模：

- `VideoREPA`: `2070` 个文件
- `videophy`: `136` 个文件

审计方法说明：

- 已经真实 clone 两个开源 repo 到 `Physical_Consistency/third_party/`
- 已对两个 repo 做完整文件树扫描
- 重点逐读了训练入口、teacher 定义、TRD 实现、评测入口、README、LingBot 本地真实训练代码
- 这份 review 是“真实源码对照审计”，不是再基于论文标题或 README 猜 API

### 0.1 最短结论

当前 `Physical_Consistency` 子项目的工程骨架是好的，隔离边界也做对了，但它还不能作为“可直接上集群、可直接得出可信 VideoREPA-style 结论”的正式实现。

更准确地说：

1. 它不是“全部凭空乱写”的。
   - `stage1_components.py` 的数据流、`prepare_y()`、`build_dataloader()`、`Wan2_1_VAE/T5/WanModel` 的调用方式，已经可以被本地真实 `train_lingbot_csgo.py` 和 `lingbot-world` 代码证实。
2. 但它也不是“已经按真实 VideoREPA / VideoPhy2 仓库严格落地”的。
   - 尤其是 `VideoMAEv2` teacher 初始化方式和 `TRD` 的实现形态，和真实 `VideoREPA` 仍然存在关键不一致。
3. 所以目前最准确的定性是：
   - 这是一个“隔离良好的 research scaffold”
   - 不是一个“已经过真实上游源码严格对齐的可发表级实现”

---

## 1. 先回答你最关心的问题

### 1.1 之前那版代码，到底是不是“直接 clone repo 后照着真实代码写的”？

不是。

但“全部都是纯猜的”这句话也不完全准确。

更准确的划分是：

1. `LingBot Stage 1` 相关调用：
   - 现在已经可以确认，大部分是参照本地真实 `train_lingbot_csgo.py` 和 `lingbot-world` API 写的，不是完全凭空猜。
2. `VideoPhy2` wrapper：
   - 之前有一部分是按公开 README 猜的。
   - 真实 clone 后确认，大部分核心调用是对的，`--batch_size` 也是真实存在的参数。
3. `VideoREPA teacher + TRD`：
   - 这一块之前明显不是基于真实 repo 严格对齐后写的。
   - 真实 clone 后已确认：路径有一部分猜对了，但核心实现细节并没有真正对上上游。

### 1.2 所以“之前集成代码的来源”应该怎么表述最准确？

建议以后统一这样表述：

> 当前 `Physical_Consistency` 子项目是一个独立隔离的研究实现。
> 其中对 LingBot Stage 1 的 continuation 路径已能被本地真实训练代码验证；
> 对 VideoPhy2 的调用方式大体正确；
> 但对 VideoREPA 的 teacher / TRD 集成此前并未基于真实上游源码完整对齐，因此目前仍应视为“VideoREPA-inspired implementation”，而不是“faithful VideoREPA port”。

---

## 2. 被真实源码证实的事实

### 2.1 `VideoREPA` 的真实情况

已经确认以下事实为真：

1. 真实 teacher 文件确实存在：
   - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/VideoREPA/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py`
2. `VideoREPA` 真实训练入口确实使用 `accelerate launch --num_processes 8`：
   - 见 `.../third_party/VideoREPA/finetune/scripts/multigpu_VideoREPA_2B_sft.sh`
3. `VideoREPA` 真实训练里确实用 `Accelerator(log_with="wandb")` 和 `accelerator.init_trackers(...)` 做 WandB：
   - 见 `.../third_party/VideoREPA/finetune/trainer.py:98-108`
   - 见 `.../third_party/VideoREPA/finetune/trainer.py:376-391`
4. `VideoREPA` 仓库自己就带了 `evaluation/VIDEOPHY2/` 和 `evaluation/videophy/` 两条评测线：
   - `.../third_party/VideoREPA/evaluation/VIDEOPHY2/eval_pipeline.sh`
   - `.../third_party/VideoREPA/evaluation/videophy/eval_pipeline.sh`

### 2.2 `VideoPhy2` 的真实情况

已经确认以下事实为真：

1. 真实 `VIDEOPHY2/inference.py` 确实存在：
   - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/videophy/VIDEOPHY2/inference.py`
2. 它真实支持这些 CLI 参数：
   - `--input_csv`
   - `--task`
   - `--checkpoint`
   - `--lora_checkpoint`
   - `--batch_size`
   - `--num_frames`
   - `--output_csv`
3. `VIDEOPHY2/README.md` 真实说明了三种输入 CSV 口径：
   - `sa`: `videopath, caption`
   - `pc`: `videopath`
   - `rule`: `rule, videopath`
4. README 明确说明输出会多一列 `score`

也就是说，之前对 `VideoPhy2` 的调用不是全错，尤其 `--batch_size` 这一条不是“猜错参数”，而是确实存在。

### 2.3 `LingBot Stage 1` 本地真实代码已经能对上的部分

这次额外核对了：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/train_lingbot_csgo.py`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/lingbot-world/wan/modules/vae2_1.py`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/lingbot-world/wan/modules/t5.py`

已经确认：

1. `Wan2_1_VAE.encode()` 真实接受 `list[[C,T,H,W]]`
2. `T5EncoderModel.__call__()` 真实返回按 token length 截断后的 context list
3. `train_lingbot_csgo.py` 真实就是 `batch_size=1` 且 `collate_fn=lambda x: x[0]`
4. `prepare_y()` 的原始 Stage 1 写法本来就是基于 `[3, F, H, W]`

因此，下面这句需要修正：

> “`video[:, 0:1]` 一定是在取 channel 维”

这在当前代码路径里并不成立。

因为当前 dataloader 明确把 batch 维剥掉了，所以：

- `video` 是 `[3, F, H, W]`
- `video[:, 0:1]` 在当前实现里确实是在取第一帧

但这不代表当前设计就没有问题，它的问题是：

- 这条训练链路被硬编码成了单样本路径
- 一旦以后要扩 batch，这套索引写法会整体失效

---

## 3. 综合结论：哪些判断被证实，哪些判断需要纠正

### 3.1 被证实的判断

以下结论在这次真实源码审计后仍然成立：

1. 当前项目结构隔离做得对
   - 所有新增内容都在 `Physical_Consistency/`
   - 没有去改共享 `code/` 目录
2. 训练启动方式有问题
   - 现在不是按你们 Stage 1 的真实 8 卡启动栈在跑
3. WandB 多进程初始化有问题
   - 如果改成 `accelerate launch`，现在会出现多进程各自 `wandb.init()`
4. 当前 trainer 的 mini-val 不符合你要求的 benchmark 口径
5. 当前 `TRD` 实现不能直接声称是“真实 VideoREPA trick”

### 3.2 需要纠正或降级的判断

以下说法在真实代码对照后需要修正：

1. “`stage1_components.py` 完全没读过真实 LingBot 代码”
   - 不成立。
   - 它和本地真实 `train_lingbot_csgo.py` 是能明显对上的。
2. “`video[:, 0:1]` 一定取错维度”
   - 在当前单样本 dataloader 设计下不成立。
3. “`VideoPhy2 --batch_size` 是瞎猜的”
   - 不成立。
   - 真实 `VIDEOPHY2/inference.py` 里确实有这个参数。

---

## 4. P0 / P1 问题总表

下面是我结合你的 review 和这次真实源码审计后，整理出的更准确版本。

### 4.1 P0-1: `VideoMAEv2Teacher` 的真实初始化参数不匹配

严重级别：

- `P0`

现状：

- 当前包装器：
  - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/teachers/videomaev2.py:21-41`
- 当前调用：
  - `factory(target_resolution=target_resolution, num_frames=num_frames)`

真实上游：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/VideoREPA/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py:330-354`
- 真实构造参数是：
  - `align_video_resolution=...`
  - `all_frames=...`

影响：

1. teacher 初始化很可能直接报错
2. 即使不报错，也不能保证初始化配置和上游一致

结论：

- 这是一个硬阻塞问题

### 4.2 P0-2: 当前 `TRD` 不是 faithful `VideoREPA` 实现

严重级别：

- `P0`

现状：

- 当前实现通过 hook 学生网络某个 block，再做自定义 token relation loss：
  - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/trd_v1.py`

真实上游：

- `VideoREPA` 的关键逻辑在：
  - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/VideoREPA/finetune/models/cogvideox_t2v_align/lora_trainer.py:176-356`
  - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/VideoREPA/finetune/models/cogvideox_t2v_align/models/cogvideox_align.py:182-238`

真实 `VideoREPA` 的关键步骤包括：

1. 不是任意 block hook，而是 transformer 显式输出 `aligns`
2. 对 teacher 输入先 `remove the first frame`
3. 把 `480x720` 缩到 `160x240`
4. 对学生特征先做 temporal upsample
5. 再通过 `downsampler_cogvideo_output` 从 `30x45` 到 `10x15`
6. 最终在对齐后的 feature volume 上做 TRD

影响：

1. 当前实验最多只能叫 `VideoREPA-inspired`
2. 不能严谨地写成“用了 VideoREPA trick”
3. 如果后面要写报告或论文，这个口径会出问题

结论：

- 如果你的目标是“先做一版不大改架构的 VideoREPA-style loss”，当前方向可以保留
- 但它必须在文档和实验命名里明确叫 `TRD-v1 / VideoREPA-inspired`
- 不能把它包装成 faithful port

### 4.3 P0-3: `metrics.item()` 会在训练中崩掉

严重级别：

- `P0`

位置：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/trd_v1.py:164`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/trd_v1.py:274-277`

问题：

- 第 `164` 行会把 `metrics` 里的所有 tensor 都 `.item()`
- 但 `274-277` 行又把 relation matrix tensor 也放进了 `metrics`

影响：

- 第一轮训练就可能抛出：
  - `Tensor cannot be converted to Scalar`

结论：

- 这是明确的运行时崩溃点

### 4.4 P0-4: WandB 会在多进程下重复初始化

严重级别：

- `P0`

位置：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/wandb_utils/session.py:14-35`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/trd_v1.py:467-483`

问题：

- `init_wandb_run()` 没有 `rank 0` 限制
- `main()` 在进入训练前直接初始化 WandB
- 如果改成 `accelerate launch`，每个进程都会执行这段代码

对照真实上游：

- `VideoREPA` 用的是 `Accelerator(log_with="wandb")` + `accelerator.init_trackers(...)`
  - `.../third_party/VideoREPA/finetune/trainer.py:98-108`
  - `.../third_party/VideoREPA/finetune/trainer.py:376-391`

影响：

1. 可能出现 8 个独立 run
2. 日志聚合会混乱
3. 你要求的“一个训练 run 看清所有失败原因”无法保证

### 4.5 P0-5: 训练脚本不是按真实 Stage 1 的 8 卡方式启动

严重级别：

- `P0`

当前脚本：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_train_trd_v1_low.sh`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/scripts/run_train_trd_v1_high.sh`

现状：

- 当前脚本是 plain `python -m ...`

真实参考：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/run_train_dual.sh`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/VideoREPA/finetune/scripts/multigpu_VideoREPA_2B_sft.sh`

影响：

1. 现在这版不等价于“接在你们真实 Stage 1 训练栈之后继续做”
2. `num_gpus: 8` / `ulysses_size: 8` 这些配置现在没有被真正消费成真实 8 卡训练

### 4.6 P0-6: 训练内验证不满足你要求的 benchmark 规范

严重级别：

- `P0`

位置：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/trd_v1.py:282-304`

问题：

- 当前 `run_light_validation()` 只计算轻量 loss
- 不生成视频
- 不走固定 manifest benchmark
- 不跑 CSGO 指标
- 不跑 VideoPhy2
- 非主进程直接 return，没有统一同步设计

影响：

1. 不满足你要求的“每 300 step mini-val + benchmark + 图表”
2. 训练失败时无法靠 WandB 复盘“物理一致性到底哪里开始坏”

### 4.7 P0-7: `TRD` 组评测存在“缺一个分支就静默回退 base”风险

严重级别：

- `P0`

位置：

- 当前评测配置：
  - `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/configs/eval_trd_v1.yaml:18-20`
- 共享评测逻辑：
  - `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/eval_batch.py:396-412`

问题：

- `eval_batch.py` 会对缺失的 `low_noise_model` / `high_noise_model` 自动回退到 `ckpt_dir`

影响：

- 如果你只训练完一个分支，评测结果可能变成：
  - `TRD-low + Base-high`
  - 或 `Base-low + TRD-high`
- 但名字上看起来还是 “TRD 组”

结论：

- 这是会污染实验结论的硬风险

---

## 5. 不是 P0，但必须明确写清楚的点

### 5.1 当前 dataloader 是单样本路径，不是通用 batch 设计

位置：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/stage1_components.py:381-388`
- 对照原始：
  - `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/train_lingbot_csgo.py:541-543`

事实：

- 当前实现和原始 Stage 1 一样，故意用了：
  - `batch_size=1`
  - `collate_fn=lambda x: x[0]`

这意味着：

1. 当前不是“frame/channel 维度已经写错”
2. 但整个 trainer 的很多写法都默认了单样本
3. 以后如果要扩 batch，`prepare_y()`、teacher encode、student reshape 等逻辑都要一起重写

### 5.2 teacher checkpoint 选择仍然不够可复现

位置：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/trainers/trd_v1.py:439-446`

问题：

- 当前是目录下 `rglob("*.pth")` 后取排序第一个

影响：

- 同一个目录里放多个 checkpoint 时，哪一个被加载取决于文件名排序

### 5.3 当前 `VideoPhy2` wrapper 基本方向是对的

位置：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/src/physical_consistency/eval/videophy2.py`

结论：

- 这一块现在不能再说是“完全乱写”
- 它和真实 `VIDEOPHY2/inference.py` 的主参数口径是对得上的

但注意：

- 它现在只是 subprocess wrapper
- 还没在你集群的真实 checkpoint / 真实生成视频上做 smoke test

---

## 6. 这套代码现在到底能不能完成你的构想

### 6.1 能完成的部分

可以完成这些：

1. 作为独立隔离子项目存在
2. 不碰共享 `code/` 主线
3. 为 `Base / Stage1 / TRD-v1` 三组实验提供独立目录、配置、lineage、脚本骨架
4. 给后续真实实现留出规范工程结构

### 6.2 还不能完成的部分

还不能直接完成这些：

1. 不能直接上集群稳定训练
2. 不能把当前结果严谨地叫做 `VideoREPA trick`
3. 不能满足你要求的“每 300 step benchmark + WandB 图表闭环”
4. 不能在不修复 P0 的前提下产出可信对比结论

### 6.3 最准确的总判断

当前状态最准确的评价是：

> 架构方向正确，工程隔离优秀，部分 LingBot/VideoPhy2 集成已经能被真实源码证实；
> 但外部 teacher 与 TRD 的关键实现还没有真正对齐到真实 VideoREPA，
> 同时训练/WandB/validation 还有多处 P0 级阻塞，
> 因此现在还不能直接作为正式实验系统使用。

---

## 7. 建议的下一步顺序

建议严格按这个顺序推进：

1. 先修 `P0-1` 到 `P0-7`
2. 明确命名口径：
   - 如果不做 faithful port，就统一叫 `TRD-v1` 或 `VideoREPA-inspired`
3. 去集群做最小 smoke test：
   - teacher 初始化
   - Stage 1 checkpoint 加载
   - 1 step forward/backward
   - WandB 单 run
4. 先做 `Base / Stage1` baseline 评测闭环
5. 再开始 `TRD-v1` continuation 训练
6. 训练稳定后再接入你要的完整 benchmark mini-val

---

## 8. 最后的明确 verdict

### 8.1 对“这个方案本身是否合理”的 verdict

合理。

把所有工作限定在：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency`

并且只读共享资产、不改共享主线，这是一个正确方案。

### 8.2 对“当前代码是否已经正确实现你的构想”的 verdict

还没有。

原因不是方向错，而是：

1. 真实 `VideoREPA` 对齐还不够
2. 训练闭环还有 P0
3. benchmark / WandB 还没达到你要求的正式标准

### 8.3 对“之前那版代码到底算什么”的最终定性

最准确的定性是：

> 不是纯瞎写的脚手架，
> 也不是已经基于真实上游源码严格落地的正式实现；
> 它是一个已经把研究边界、工程结构、LingBot continuation 骨架搭好的独立子项目，
> 但要真正变成你要的物理一致性实验系统，还必须先完成一轮真实源码对齐和 P0 修复。
