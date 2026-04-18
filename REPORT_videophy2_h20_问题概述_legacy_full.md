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
