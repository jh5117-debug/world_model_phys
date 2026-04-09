# LingBot CSGO 物理一致性工作流 v1

本文档用于规划一条“不破坏现有代码主线”的物理一致性实验路线。

核心目标：

1. 先在现有 Stage 1 基础上建立一条可重复的物理一致性评测链路。
2. 对比三组模型：
   - Base LingBot
   - CSGO SFT 后的 LingBot
   - CSGO SFT + VideoREPA-style trick
3. 主 benchmark 优先使用 VideoPhy-2。
4. 同时保留 CSGO 域内指标，避免只看开域 benchmark。
5. 所有新增内容都放在 `Physical_Consistency/` 下，原始代码尽量不改；大文件只做软链接。
6. 新增代码必须符合正规 project 结构，不写成零散脚本堆。
7. 所有权重、数据、输出路径都必须支持命令行覆盖，默认值统一写成集群路径。

---

## 0A. 本轮新增硬约束

这部分是本轮补充的强约束，后续实现代码时必须严格遵守。

### 0A.0 工作边界法则

我只保证并且只会做这些事：

1. 只在 `Physical_Consistency` 下面新增代码、配置、脚本和文档。
2. 不修改 `code` 下面任何现有训练/评测代码。
3. 不为别的方向建立目录、脚本、配置或输出。
4. 共享资产只读不写：只读取 `Stage 1 checkpoint`、`processed_csgo_v3`、`raw_csgo_v3`。
5. 实验输出单独落到 `physical_consistency` 命名空间，不覆盖别人结果。
6. 如果后面某一步真的必须碰共享代码，我会先停下来明确说明，不会默认去改。

### 0A.1 工程结构必须正规化

`Physical_Consistency/` 不能只是一个“脚本堆目录”，而要按一个独立子项目来组织。

最低要求：

1. 必须有 `pyproject.toml`
2. 必须有明确的 `src/physical_consistency/` 包目录
3. 必须有 `configs/`
4. 必须有 `scripts/`
5. 必须有 `tests/`
6. 训练、评测、数据准备逻辑必须模块化，不允许把所有逻辑塞进一个超长脚本

### 0A.2 路径必须命令行可控

因为你当前本地没有数据集和权重，所以所有涉及路径的代码都必须这样设计：

1. 每个训练/评测脚本都提供命令行参数
2. 默认值直接写成集群上的真实路径
3. 运行时允许 `--xxx_path ...` 覆盖
4. 不允许把“本地调试路径”写死进代码

例如未来训练脚本必须类似这样设计：

- `--base_model_dir` 默认 `/home/nvme02/lingbot-world/models/lingbot-world-base-act`
- `--stage1_ckpt_dir` 默认 `/home/nvme02/lingbot-world/output/dual_ft_v3/epoch_2`
- `--dataset_dir` 默认 `/home/nvme02/lingbot-world/datasets/processed_csgo_v3`
- `--raw_data_dir` 默认 `/home/nvme02/lingbot-world/datasets/raw_csgo_v3/dust2-80-32fps/aef9560bbce0c405/e61e20c503eb4af78d4b2011f945aca0/train`

### 0A.3 默认口径必须是“基于当前 Stage 1 之后继续做”

如果你要证明这条线是“在当前 SFT 之后继续做的物理一致性工作”，那默认训练入口就必须满足：

1. 训练初始化默认从 `Stage 1 checkpoint` 读，而不是从 base model 直接读
2. 输出目录里必须保存 lineage 信息
3. 脚本启动时必须打印 `parent_stage1_ckpt`
4. 评测汇总里必须写明当前实验的 parent checkpoint

换句话说：

- `Base` 只用于 zero-shot baseline
- 真正的物理一致性训练默认父模型必须是 `Stage 1`

### 0A.4 范围必须严格限定在 `Physical_Consistency`

本项目只负责：

1. 基于当前 `Stage 1` checkpoint 的物理一致性增强
2. 物理一致性 benchmark 与分析
3. `Physical_Consistency/` 目录内的新代码、新配置、新输出

本项目明确不负责：

1. `stage2` 多视角模块
2. `long_memory` 相关模块
3. 为其他组建立任何目录、脚本、配置或输出规划

因此：

把当前工作完整放进 `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency`，
并且只在这个子项目内部扩展，是本阶段的唯一推荐方案。

---

## 0. 结论先行

你的思路是合理的，但要加两条约束：

1. `VideoPhy-2` 适合作为主 benchmark，但不能作为唯一 benchmark。
   原因：它是开域动作物理评测，能反映“视频是否更物理”，但不一定能直接抓住 CSGO 第一人称里的“队友飘着走、墙体漂移、箱子无外力移动”这类游戏域错误。
2. 第一阶段严格只做 `Physical_Consistency`，不把其他方向的模块混进来。
   原因：当前目标是先确认“Stage 1 之后的物理一致性增强”是否成立，范围必须收紧。

因此本阶段固定为：

- 主线：`Base` vs `Stage1-SFT(epoch_2)` vs `Stage1-SFT+TRD-v1`
- 主 benchmark：`VideoPhy-2 AutoEval`
- 辅助 benchmark：现有 CSGO 指标 + 小规模人工审查
- 不引入任何 `stage2` 或 `long_memory` 内容

---

## 1. 已确认的现有代码入口

现有主仓库路径：

- `/home/hj/Multi-View-Physically-Consistent-World-Model`

现有 Stage 1 训练入口：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/train_lingbot_csgo.py`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/run_train_dual.sh`

现有 Stage 1 评估入口：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/eval_batch.py`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/eval_fid_fvd.py`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/eval_action_control.py`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/run_eval_ablation.sh`
- `/home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune/aggregate_ablation.py`

本轮新工作目录：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency`

注意：

- 我当前这个本地环境下看不到 `/home/nvme02/lingbot-world/...`。
- 所以下面所有指向 `/home/nvme02/lingbot-world/...` 的软链接步骤，都默认是在你的集群环境执行。

---

## 2. 新目录规划

后续所有新增文件统一放在这里：

```text
/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency
├── README_physical_consistency_workflow_v1.md
├── pyproject.toml
├── .gitignore
├── Makefile
├── configs
│   ├── path_config_cluster.env
│   ├── eval_base.yaml
│   ├── eval_stage1_epoch2.yaml
│   ├── train_trd_v1.yaml
│   ├── eval_trd_v1.yaml
│   └── videophy2_eval.yaml
├── links
│   ├── base_model
│   ├── stage1_epoch2
│   ├── stage1_final
│   ├── processed_csgo_v3
│   ├── raw_csgo_v3_train
│   ├── lingbot_code
│   └── finetune_code
├── third_party
│   ├── VideoREPA
│   └── videophy
├── data
│   ├── manifests
│   │   ├── csgo_phys_val50.csv
│   │   ├── csgo_phys_val200.csv
│   │   └── videophy2_inputs
│   ├── audit
│   │   ├── csgo_physics_taxonomy_v1.md
│   │   └── csgo_human_eval_template.csv
│   └── cache
├── src
│   └── physical_consistency
│       ├── __init__.py
│       ├── common
│       ├── datasets
│       ├── teachers
│       ├── losses
│       ├── trainers
│       ├── eval
│       ├── lineage
│       └── wandb_utils
├── scripts
│   ├── setup_symlinks.sh
│   ├── setup_videorepa.sh
│   ├── setup_videophy2.sh
│   ├── build_fixed_val_sets.py
│   ├── verify_stage1_lineage.py
│   ├── run_eval_base.sh
│   ├── run_eval_stage1_epoch2.sh
│   ├── run_train_trd_v1_low.sh
│   ├── run_train_trd_v1_high.sh
│   ├── run_eval_trd_v1.sh
│   ├── run_videophy2_all.sh
│   ├── run_csgo_metrics_all.sh
│   └── run_compare_all.sh
├── tests
│   ├── test_path_resolution.py
│   ├── test_lineage_contract.py
│   └── test_manifest_builder.py
├── runs
│   ├── wandb
│   ├── generated_videos
│   ├── eval
│   ├── reports
│   └── checkpoints
└── logs
```

原则：

1. 原仓库里的 `code/finetune_v3/lingbot-csgo-finetune/` 不直接改。
2. 所有新实验脚本都在 `Physical_Consistency/` 下新建。
3. 大权重、大数据、大输出都放集群路径，并通过软链接或配置指向。
4. 新代码统一通过 `src/physical_consistency/` 提供模块，不写脏乱的相对导入。
5. `scripts/` 只做薄封装，核心逻辑必须在包内。

---

## 2A. 工程实现规范

后续代码实现时，采用以下规范。

### 2A.1 Python 包组织

1. 包名固定为 `physical_consistency`
2. 核心训练逻辑进入：
   - `src/physical_consistency/trainers/`
3. teacher 封装进入：
   - `src/physical_consistency/teachers/`
4. loss 进入：
   - `src/physical_consistency/losses/`
5. eval 进入：
   - `src/physical_consistency/eval/`
6. WandB 相关统一进入：
   - `src/physical_consistency/wandb_utils/`

### 2A.2 代码风格

1. 所有模块必须有清晰 docstring
2. 所有公开函数必须带类型注解
3. 配置项统一走 `argparse + yaml/env`
4. 不允许到处散落 hard-code path
5. 日志统一走 `logging`
6. 输出文件名、目录名、run 名必须稳定可复现

### 2A.3 脚本职责

1. `scripts/*.sh`
   - 只负责环境变量、参数拼接、启动命令
2. `scripts/*.py`
   - 只负责很薄的 CLI 入口或一次性工具
3. 真正训练和评测逻辑
   - 必须在 `src/physical_consistency/` 里

---

## 2B. 路径与默认值规范

后续所有脚本都要遵守下面这个优先级：

1. 命令行显式传参
2. `configs/path_config_cluster.env`
3. 代码中的默认值

而代码中的默认值必须写成集群路径，不写本地空路径。

推荐默认值：

- `BASE_MODEL_DIR=/home/nvme02/lingbot-world/models/lingbot-world-base-act`
- `STAGE1_CKPT_DIR=/home/nvme02/lingbot-world/output/dual_ft_v3/epoch_2`
- `STAGE1_FINAL_DIR=/home/nvme02/lingbot-world/output/dual_ft_v3/final`
- `DATASET_DIR=/home/nvme02/lingbot-world/datasets/processed_csgo_v3`
- `RAW_DATA_DIR=/home/nvme02/lingbot-world/datasets/raw_csgo_v3/dust2-80-32fps/aef9560bbce0c405/e61e20c503eb4af78d4b2011f945aca0/train`
- `WANDB_DIR=/home/nvme02/lingbot-world/output/wandb`
- `OUTPUT_ROOT=/home/nvme02/lingbot-world/output/physical_consistency`

注意：

这里的默认值是“真实运行默认值”，不是示例值。
也就是说，未来就算你在本地打开代码，代码里看到的默认值也应该是集群上的真实地址。

---

## 3. 集群侧软链接规划

在集群上执行，目标是把所有外部依赖统一映射到 `Physical_Consistency/links/`。

建议软链接目标：

- `links/base_model -> /home/nvme02/lingbot-world/models/lingbot-world-base-act`
- `links/stage1_epoch2 -> /home/nvme02/lingbot-world/output/dual_ft_v3/epoch_2`
- `links/stage1_final -> /home/nvme02/lingbot-world/output/dual_ft_v3/final`
- `links/processed_csgo_v3 -> /home/nvme02/lingbot-world/datasets/processed_csgo_v3`
- `links/raw_csgo_v3_train -> /home/nvme02/lingbot-world/datasets/raw_csgo_v3/dust2-80-32fps/aef9560bbce0c405/e61e20c503eb4af78d4b2011f945aca0/train`
- `links/lingbot_code -> /home/hj/Multi-View-Physically-Consistent-World-Model/code/lingbot-world`
- `links/finetune_code -> /home/hj/Multi-View-Physically-Consistent-World-Model/code/finetune_v3/lingbot-csgo-finetune`

建议在 `scripts/setup_symlinks.sh` 中只做三件事：

1. 检查源路径是否存在。
2. 若目标软链接已存在则不覆盖。
3. 打印最终映射表到 `logs/setup_symlinks.log`。

---

## 3A. 如何证明这条线是“Stage 1 之后继续做”的

这是很关键的问题。

如果所有代码都放在 `Physical_Consistency/`，但你想证明它不是一个脱离主线的旁支实验，而是“基于当前 SFT 后 Stage 1 的进一步工作”，就必须建立一个明确的 lineage contract。

### 3A.1 训练初始化约束

物理一致性训练脚本默认必须使用：

- `--stage1_ckpt_dir /home/nvme02/lingbot-world/output/dual_ft_v3/epoch_2`

而不是：

- `--base_model_dir /home/nvme02/lingbot-world/models/lingbot-world-base-act`

`base_model_dir` 仅用于：

1. zero-shot baseline 评测
2. 回填 T5/VAE 等公共资产

### 3A.2 输出目录必须保存 lineage 元数据

每次训练输出目录里必须保存：

- `lineage.json`

至少包含：

1. `experiment_group = physical_consistency`
2. `parent_stage = stage1`
3. `parent_stage1_ckpt`
4. `base_model_dir`
5. `dataset_dir`
6. `git_commit`
7. `config_path`
8. `config_hash`
9. `created_at`

推荐输出位置：

- `runs/checkpoints/<exp_name>/lineage.json`

### 3A.3 启动时强校验

训练脚本启动后必须先检查：

1. `stage1_ckpt_dir` 是否存在
2. 里面是否同时有 `low_noise_model/` 和 `high_noise_model/`
3. 若缺失则直接报错退出

建议额外提供：

- `scripts/verify_stage1_lineage.py`

用于在训练前和汇报前校验 lineage。

### 3A.4 报告层必须明确 parent

你后面每一份汇总表、每一个 WandB run、每一个 checkpoint 名字都要明确写出：

- `parent=stage1_epoch2`

这样最后任何人看实验结果，都能明确知道：

这不是从 base 直接另起炉灶，而是从当前 Stage 1 SFT 继续做的物理一致性增强。

---

## 3B. 为什么把工作放在 `Physical_Consistency/` 是合理方案

答案是：是的，这可以作为一个正式解决方案，而且很适合当前三组并行开发状态。

前提是你把它定义成：

一个“基于 Stage 1 的独立研究子项目”，而不是“绕开主仓库的私人脚本目录”。

它的优点是：

1. 不污染主线代码
2. 便于单独迭代、单独回滚、单独汇报
3. 后续如果结果好，再选择性合并回主线

因此推荐定位是：

- `Physical_Consistency = Stage 1 之后的物理一致性扩展层`

而不是：

- `/home/nvme02/lingbot-world/output/physical_consistency/*`
- `Physical_Consistency = 替代现有 Stage 1 的新主线`

---

## 3C. 范围边界

这份文档只定义 `Physical_Consistency` 子项目本身。

允许引用的共享资产只有：

1. `Stage 1 checkpoint`
2. `processed_csgo_v3`
3. `raw_csgo_v3`
4. 公共 benchmark manifest

这份文档不为任何其他方向建立：

1. 目录结构
2. 输出目录
3. 配置文件
4. 训练脚本
5. 评测脚本

---

## 4. 外部开源仓库接入

### 4.1 VideoREPA

克隆到：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/VideoREPA`

用途：

1. 参考 TRD loss 设计。
2. 参考 teacher feature 抽取方式。
3. 不直接把它的 CogVideoX 训练框架搬进 LingBot 主线。

本项目里的使用原则：

- 只借鉴 `VideoREPA-style auxiliary loss`
- 不照搬它的完整生成框架
- 不改 LingBot 主干架构太多

### 4.2 VideoPhy-2

克隆到：

- `/home/hj/Multi-View-Physically-Consistent-World-Model/Physical_Consistency/third_party/videophy`

用途：

1. 使用 `VIDEOPHY2/inference.py`
2. 下载 `VideoPhy-2 AutoEval` checkpoint
3. 对三组模型生成视频做 `SA`、`PC`、`Rule` 打分

---

## 5. 固定实验设计

本阶段先只比较三组：

1. `exp_base_zeroshot`
   - 权重：`links/base_model`
2. `exp_stage1_epoch2`
   - 权重：`links/stage1_epoch2`
3. `exp_stage1_epoch2_trd_v1`
   - 权重：未来训练得到的新 checkpoint

所有对比都必须固定以下项：

1. 相同 val clip 集合
2. 相同 prompt
3. 相同 first frame
4. 相同 action control
5. 相同采样参数
6. 相同随机种子集合

推荐种子：

- `42`
- `123`
- `3407`

推荐采样参数：

- `frame_num = 81`
- `sample_steps = 70`
- `guide_scale = 5.0`
- `height = 480`
- `width = 832`

原因：

1. 你们现有 `eval_batch.py` 默认口径就是这一组参数。
2. 视频生成有随机性，单 seed 容易误判。
3. 至少要做 `3 seeds` 平均，才能把“某次生成刚好不飞”这种偶然性压下去。

---

## 6. 评测分三层

### 6.1 层 1：VideoPhy-2 主 benchmark

目标：

- 判断生成视频是否更符合物理常识

输入形式：

1. `SA` 任务：`videopath + caption`
2. `PC` 任务：`videopath`
3. `Rule` 任务：`rule + videopath`

在你的项目里：

1. `SA` 使用现有 CSGO prompt。
2. `PC` 对所有生成视频都跑。
3. `Rule` 不直接用原始 VideoPhy-2 rules，第一版先不强行套；先把 `SA/PC/joint` 跑通。

输出目录建议：

- `runs/eval/videophy2/base_seed42`
- `runs/eval/videophy2/stage1_epoch2_seed42`
- `runs/eval/videophy2/stage1_trd_v1_seed42`

最终汇总字段：

1. `SA_mean`
2. `PC_mean`
3. `Joint(SA>=4 & PC>=4)`
4. `seed_mean`
5. `seed_std`

### 6.2 层 2：现有 CSGO 指标

当前直接可复用：

1. `eval_batch.py`
   - `PSNR`
   - `SSIM`
   - `LPIPS`
2. `eval_fid_fvd.py`
   - `FID`
   - `FVD`
3. `eval_action_control.py`
   - 动作控制一致性

### 6.3 层 3：CSGO 物理异常人工审查

这是必须加的一层，因为它最接近你真正关心的问题。

建议做一个 `50 clip x 3 model x 3 seeds` 的小审查表，类别固定为：

1. `self_flying`
2. `teammate_flying`
3. `static_wall_motion`
4. `static_box_motion`
5. `camera_teleport_or_snap`
6. `scale_or_shape_instability`
7. `other_physics_violation`

建议文件：

- `data/audit/csgo_physics_taxonomy_v1.md`
- `data/audit/csgo_human_eval_template.csv`

这样你最后能回答三件事：

1. VideoPhy-2 上是否更物理
2. CSGO 域内像素/动作指标是否退化
3. 你最关心的“飞起来、墙在动、箱子漂移”是否真的下降

---

## 7. 先做基线，不改架构

第一阶段不训练新模型，先跑完以下 baseline：

### 7.1 Base LingBot

用现有评估脚本直接跑：

- base model
- 固定 val subset
- 3 seeds

结果保存到：

- `runs/eval/base/...`

### 7.2 CSGO SFT epoch_2

用同样口径跑：

- `links/stage1_epoch2`
- 同一 val subset
- 同一 seeds

结果保存到：

- `runs/eval/stage1_epoch2/...`

### 7.3 基线结论门槛

只有在以下条件成立时，才进入 `TRD-v1`：

1. `stage1_epoch2` 在 CSGO 域内指标上不显著差于 base
2. 你确认它确实存在明显物理错误
3. 这些错误在人工审查中有足够样本出现

如果连 baseline 差异都不稳定，就先不该训练 trick。

---

## 8. TRD-v1 的最小可行方案

目标：

- 做一版“不大改 LingBot 架构”的 VideoREPA-style loss

第一版只加 loss，不加新控制分支，不改采样器，不动 Stage 2。

### 8.1 teacher 选择

第一版 teacher：

- `VideoMAEv2` frozen

放置位置建议：

- `Physical_Consistency/third_party/VideoREPA/ckpt/VideoMAEv2/...`
或
- 集群大盘路径后软链接到 `Physical_Consistency/links/teacher_videomaev2`

### 8.2 student feature 选择

student 不是输出视频，而是 LingBot DiT 中间层 token。

第一版建议：

1. 从 `low_noise_model` / `high_noise_model` 的中后层抽取一个 block 的 token
2. 用一个轻量 projector 把 student token 映射到 teacher 对齐维度
3. 在 token relation 上做 loss，而不是 token value 硬对齐

### 8.3 损失函数

总损失：

`L_total = L_fm + lambda_trd * (lambda_s * L_spatial + lambda_t * L_temporal)`

第一版推荐：

- `lambda_trd = 0.1`
- `lambda_s = 1.0`
- `lambda_t = 1.0`

第一版只做：

1. relation distillation
2. 单层 student feature
3. 单个 teacher

第一版不做：

1. 多 teacher
2. 多层蒸馏
3. 额外重建 loss
4. 改 Stage 2

### 8.4 训练策略

沿用现有 Stage 1 双模型拆分训练：

1. `run_train_trd_v1_low.sh`
   - 训练 `low_noise_model + TRD`
2. `run_train_trd_v1_high.sh`
   - 训练 `high_noise_model + TRD`

这样对现有训练代码侵入最小，也方便与你已有 `epoch_2` 对齐。

---

## 9. WandB 强制规范

你提的 WandB 要求必须作为硬规则执行。

### 9.1 初始化时机

未来训练入口脚本里，`wandb.init()` 必须放在最前面，至少要早于：

1. 模型加载
2. 数据集构建
3. accelerate 初始化之后的大部分重逻辑

推荐顺序：

1. parse args
2. 组装 run name / config
3. `wandb.init(...)`
4. 再进入训练逻辑

这样即便后面包报错，run 也已经建起来了。

### 9.2 每次训练必须记录的标量

1. `train/loss_total`
2. `train/loss_fm`
3. `train/loss_trd`
4. `train/loss_trd_spatial`
5. `train/loss_trd_temporal`
6. `train/lr`
7. `train/grad_norm`
8. `train/global_step`
9. `train/epoch`
10. `train/sample_sigma`
11. `train/sample_timestep`
12. `train/teacher_feat_norm`
13. `train/student_feat_norm`
14. `train/pred_target_cosine`

### 9.3 每次训练必须记录的图表/媒体

1. loss 曲线
2. lr 曲线
3. grad norm 曲线
4. timestep/sigma 分布直方图
5. spatial relation matrix 可视化
6. temporal relation matrix 可视化
7. 固定 val clip 的 GT / Gen 对比视频
8. 固定 val clip 的首帧、末帧拼图
9. 每次验证的指标表

### 9.4 验证频率

推荐两级验证：

1. `every 300 optimizer steps`
   - 跑一个 `mini-val`
   - 8 个固定 clip
   - 1 个 seed
   - 指标：`PSNR/SSIM/LPIPS/action_control/VideoPhy2-PC`
2. `every epoch end`
   - 跑 `full-val`
   - 50 个固定 clip
   - 3 个 seeds
   - 指标：全部汇总

### 9.5 WandB artifact

每个 epoch 至少上传：

1. checkpoint 路径
2. mini-val report
3. full-val report
4. 代表性生成视频
5. 失败样例视频

---

## 10. 推荐执行顺序

### Phase A：只做评测闭环

1. 建好 `Physical_Consistency/` 目录和软链接
2. 下载 `VideoREPA` 与 `VideoPhy-2`
3. 固定 `val50` 和 `val200` 两个 manifest
4. 跑 `Base`
5. 跑 `Stage1 epoch_2`
6. 汇总 `VideoPhy-2 + CSGO 指标 + 人工审查`

输出：

- 一份 baseline 对比表
- 一份错误 taxonomy
- 一批固定失败样例

### Phase B：实现 TRD-v1

1. 新建独立 trainer，不改原 `train_lingbot_csgo.py`
2. 复用原数据读取逻辑
3. 增加 teacher feature 抽取
4. 增加 TRD loss
5. 增加完整 WandB
6. 增加 mini-val / epoch-val

### Phase C：训练并比较

1. 先训练 `low`
2. 再训练 `high`
3. 跑三组统一 benchmark
4. 选最优 checkpoint

---

## 11. 这阶段不要做的事

1. 不要一开始就把 Stage 2 加进主实验
2. 不要同时尝试多个 physics trick
3. 不要只看单 seed
4. 不要只看 VideoPhy-2，不看 CSGO 人工异常
5. 不要直接改原始共享训练脚本

---

## 12. 第一轮产出物清单

第一轮完成后，至少应该得到：

1. `Base` 的全套评测结果
2. `Stage1 epoch_2` 的全套评测结果
3. 固定 `val50` / `val200` manifest
4. CSGO 物理异常 taxonomy
5. `TRD-v1` 训练设计文档
6. 明确的 WandB 字段规范

只有这些都齐了，才进入代码实现。

---

## 13. 外部参考

外部方法和 benchmark 参考来源：

- VideoREPA project page: `https://videorepa.github.io/`
- VideoREPA official code: `https://github.com/aHapBean/VideoREPA`
- VideoPhy / VideoPhy-2 official repo: `https://github.com/Hritikbansal/videophy`
- VideoPhy-2 page: `https://videophy2.github.io/`
