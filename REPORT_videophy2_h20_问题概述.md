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

## 10. 当前最重要的实验结论

### 10.1 当前 AutoEval 结果不可信

截至目前，最稳妥的结论是：

- 当前 H20 上的 `VideoPhy-2-AutoEval` 不能作为最终实验结论依据

原因不是单一的，而是以下证据共同支持：

- 早期全 `0` 实际来自空输出解析失败
- 中期全 `1` 伴随明显坏输出
- 严格 parser 之后，单样本 `SA/PC` 都暴露为 `'100000'` 且 `score` 留空
- 额外自然视频 sanity check 也没有恢复区分能力

### 10.2 当前系统更像“坏输出生成器”，不是“有效 judge”

到这一步，当前最准确的表述不是：

- “VideoPhy-2 认为所有视频得分都很低”

而是：

- “VideoPhy-2 在当前环境与配置下没有稳定地产生可信的评分答案”

### 10.3 不能再把当前 AutoEval 当主指标

因此，在论文、汇报或实验记录里，当前 VideoPhy-2-AutoEval 应该被描述为：

- sanity check failed
- collapsed judge
- currently not reliable on H20 under our setup

而不应继续被拿来作为：

- 主结果表格中的正式 `SA/PC/joint`

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

## 15. 当前最合理的结论写法

如果要把当前阶段写进报告或论文内部记录，建议使用下面这段结论。

### 15.1 可直接引用的中文结论

在 H20 环境中，我们成功完成了 VideoPhy-2-AutoEval 所需的权重部署、依赖修复、脚本并行化与日志化改造，并使评测链路可以稳定执行。然而，进一步的单样本诊断、全量 `CS:GO val` 评测以及额外自然视频 sanity check 表明，当前 `videophy_2_auto` 在我们的环境与配置下并未产生具有区分能力的有效评分输出。其失败形式依次表现为：空输出导致的假 `0`、坏输出经宽松 parser 折叠后的假 `1`，以及在严格解析下暴露出的重复坏串（如 `'100000'`）。因此，截至目前，VideoPhy-2-AutoEval 在 H20 上不能作为 LingBot / LingBot-Stage1 的最终物理一致性评测指标。后续若要报告标准 VideoPhy-2 benchmark 结果，必须先实现 benchmark-safe 的 LingBot inference mode，并重新验证 judge 的有效性；否则只能将当前实验归类为内部 conditioned evaluation，而不能视为标准 VideoPhy-2 benchmark。

## 16. 建议的下一步

### 16.1 不要再把当前 AutoEval 作为主结果

短期内建议：

- 停止把当前 `VideoPhy-2-AutoEval` 数字写入主结果表
- 仅保留为失败分析记录

### 16.2 优先实现 benchmark-safe inference mode

如果要继续往标准 benchmark 推进，下一步最重要的是：

- 给 LingBot / LingBot-Stage1 设计一个只依赖 `prompt + seed + benchmark-safe defaults` 的推理模式

在这个模式下，不能再使用：

- GT first frame
- GT poses
- GT actions
- GT intrinsics

### 16.3 评测建议

在 benchmark-safe inference mode 尚未完成前，推荐：

- 人工评测优先
- 或改用已验证的替代评测方案

而不是继续依赖当前已经退化的 VideoPhy-2-AutoEval。

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

## 18. 最终一句话总结

截至目前，`VideoPhy-2-AutoEval` 在 H20 上对我们的任务不是一个可信的最终 judge；而 LingBot / LingBot-Stage1 由于推理输入不止 prompt，本身也不能在当前形态下直接声称是标准 VideoPhy-2 benchmark。后续如果要做正式 benchmark，必须先把“模型输入协议”和“评测协议”同时收敛到 benchmark-safe 版本。

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
