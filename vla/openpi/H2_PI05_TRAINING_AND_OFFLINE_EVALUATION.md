# H2 pi0.5 训练与离线动作验证指南

本文档说明如何在本仓库中，基于 `pi05_base` 微调 Unitree H2 单手操作数据，并使用记录数据离线检查模型预测动作。所有命令均在仓库根目录执行。

## 1. 当前配置

训练配置名为 `pi05_h2_one_hand`，定义在 `src/openpi/training/config.py`。

新数据目录 `data_train/lerobot_h2_one_hand_15hz_8_10_mixu` 对应独立配置
`pi05_h2_one_hand_mixu`。它使用单独的归一化统计目录和 checkpoint 目录，不会覆盖原始
`pi05_h2_one_hand`（8_9 数据）训练结果。

| 项目 | 当前值 |
| --- | --- |
| 基础模型 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| 微调方式 | LoRA，仅训练 LoRA 适配器 |
| 本地数据目录 | `data_train/lerobot_h2_one_hand_15hz_8_9` |
| LeRobot 元数据 repo ID | `liguopu/h2_one_hand` |
| 数据格式 | LeRobot v3，50 episodes / 17,485 frames / 15 Hz |
| 相机 | 头部、左腕、右腕三路 RGB 视频 |
| state/action | 29 维 float32 绝对关节角，单位 rad |
| 模型动作维度 | 32 |
| 动作块长度 | 50 steps，约 3.33 s |
| batch size | 1 |
| 训练步数 | 20,000 |
| checkpoint 间隔 | 1,000 steps |

新数据配置的本地数据目录为 `data_train/lerobot_h2_one_hand_15hz_8_10_mixu`，归一化统计会写入：

```text
assets/pi05_h2_one_hand_mixu/liguopu/h2_one_hand_8_10_mixu/
```

`state` 和 `action` 的 29 个维度会由 `PadStatesAndActions` 自动在末尾补零至 32 维。H2 配置不会应用 `DeltaActions`，训练目标和模型输出均保留绝对关节位置语义。

## 2. 环境与离线模式

使用仓库的 `uv` 环境：

```bash
uv sync
uv run python --version
```

数据和已下载 checkpoint 都在本地时，建议启用 Hugging Face 离线模式，避免数据 loader 访问网络：

```bash
export HF_HUB_OFFLINE=1
```

也可以把 `HF_HUB_OFFLINE=1` 放在单条命令前面。首次训练需要能访问或已缓存 `pi05_base` 权重；之后训练和离线检查均可在离线模式运行。

## 3. 检查数据与配置

先确认训练配置可以解析：

```bash
HF_HUB_OFFLINE=1 uv run python - <<'PY'
from openpi.training.config import get_config

config = get_config("pi05_h2_one_hand")
data = config.data.create(config.assets_dirs, config.model)
print("model:", config.model.model_type.value)
print("action horizon:", config.model.action_horizon)
print("batch size:", config.batch_size)
print("repo id:", data.repo_id)
print("data root:", data.data_root)
PY
```

预期关键输出：

```text
model: pi05
action horizon: 50
batch size: 1
repo id: liguopu/h2_one_hand
data root: data_train/lerobot_h2_one_hand_15hz_8_9
```

本仓库锁定的 LeRobot 版本原生读取 v2 数据格式。H2 数据为 LeRobot v3，因此 `src/openpi/training/data_loader.py` 中的本地 v3 适配器会读取 parquet 元数据、帧数据与视频文件；不需要将数据转换成旧版格式。

## 4. 计算归一化统计

首次训练前必须为当前数据生成完整的 state/action 统计：

1-单手操作
```bash
HF_HUB_OFFLINE=1 uv run python scripts/compute_norm_stats.py \
  --config-name pi05_h2_one_hand
```

2-双手操作
```bash
HF_HUB_OFFLINE=1 uv run python scripts/compute_norm_stats.py \
  --config-name pi05_h2_two_hand
```

统计文件写入：

```text
assets/pi05_h2_one_hand/liguopu/h2_one_hand/
```

检查命令结束时应出现类似输出：

```text
Writing stats to: .../assets/pi05_h2_one_hand/liguopu/h2_one_hand
```

不要使用带 `--max-frames` 的统计结果开始正式训练；该选项只适用于快速检查数据管线。若曾生成小样本统计，请删除整个 `assets/pi05_h2_one_hand` 目录后重新计算完整统计。

可额外检查数据关节范围。当前数据的观测范围约为 $[-1.3233, 1.4250]$ rad，动作范围约为 $[-1.3435, 1.4250]$ rad，均在合理范围内：

```bash
uv run python - <<'PY'
import numpy as np
import polars as pl

path = "data_train/lerobot_h2_one_hand_15hz_8_9/data/chunk-000/file-000.parquet"
data = pl.read_parquet(path, columns=["observation.state", "action"])
for key in ("observation.state", "action"):
    values = np.asarray(data[key].to_list(), dtype=np.float32)
    print(f"{key}: min={values.min():.6f}, max={values.max():.6f}")
PY
```

## 5. 开始训练

使用唯一的实验名启动训练：

### 新 8_10_mixu 单手数据：计算统计后立即训练

以下是一条完整指令：仅当归一化统计成功写入后，`&&` 才会开始训练。实验输出目录为
`../../weights/openpi/checkpoints/pi05_h2_one_hand_mixu/pi05_h2_one_hand/`。

```bash
HF_HUB_OFFLINE=1 uv run python scripts/compute_norm_stats.py \
  --config-name pi05_h2_one_hand_mixu && \
HF_HUB_OFFLINE=1 uv run python scripts/train.py \
  pi05_h2_one_hand_mixu \
  --exp-name pi05_h2_one_hand
```

h2_dianzha_down_no_hand 电闸
```bash
HF_HUB_OFFLINE=1 uv run python scripts/compute_norm_stats.py \
  --config-name h2_dianzha_down_no_hand && \
HF_HUB_OFFLINE=1 uv run python scripts/train.py \
  h2_dianzha_down_no_hand \
  --exp-name pi05_h2_dianzha_down_no_hand
```

1-单手操作
```bash
HF_HUB_OFFLINE=1 uv run python scripts/train.py \
  pi05_h2_one_hand \
  --exp-name smoke_test
```
2-双手操作
```bash
HF_HUB_OFFLINE=1 uv run python scripts/train.py \
  pi05_h2_two_hand \
  --exp-name two_hand_v1
```


默认输出目录为：

```text
../../weights/openpi/checkpoints/pi05_h2_one_hand/smoke_test/
```

最终 20,000 steps 正常情况下会保存为 `19999`，因为训练 step 从 0 开始编号：

```text
../../weights/openpi/checkpoints/pi05_h2_one_hand/smoke_test/19999/
```

成功结束时应看到：

```text
20000/20000
Finished saving checkpoint
No errors found in background save thread
Done waiting for Save Finalize thread
```

检查最终 checkpoint 内容：

```bash
find ../../weights/openpi/checkpoints/pi05_h2_one_hand/smoke_test/19999 -maxdepth 1 -type d | sort
```

应至少包含 `assets`、`params` 和 `train_state`。

### 恢复中断训练

训练中断且要从最新 checkpoint 继续时，使用同一实验名并加 `--resume`：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/train.py \
  pi05_h2_one_hand \
  --exp-name smoke_test \
  --resume
```

不要同时使用 `--resume` 和 `--overwrite`。

## 6. 离线动作预测验证

离线检查脚本为 `scripts/check_h2_offline_predictions.py`。它会：

1. 从本地 H2 数据集取三路图像、29 维 state 和 task 文本。
2. 加载指定 checkpoint 内保存的归一化统计和模型参数。
3. 预测 50 步动作块。
4. 截取预测的前 29 维，与记录的 50 x 29 绝对动作标签比较。
5. 保存数值指标、原始动作块和图片，不向机器人发送任何命令。

运行最终 checkpoint 的默认样本集：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions.py
```

通过 `--config-name` 选择要验证的 H2 任务。脚本默认读取
`../../weights/openpi/checkpoints/<config-name>/smoke_test/19999`，并写入
`visualization_output/<config-name>_offline_predictions/`；可用
`--checkpoint-dir` 和 `--output-dir` 覆盖这两个路径。

```bash
# 单手任务
HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions.py \
  --config-name pi05_h2_one_hand

# 双手任务
HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions.py \
  --config-name pi05_h2_two_hand
```

默认抽样索引为 `0 1000 5000 10000 15000`。输出目录为：

```text
visualization_output/pi05_h2_one_hand_offline_predictions/
```

指定单手 checkpoint、样本和独立输出目录，例如比较 step 19999：

14 DOF
```bash
HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions_14dof.py \
  --config-name h2_dianzha_down_no_hand \
  --checkpoint-dir ../../weights/openpi/checkpoints/h2_dianzha_down_no_hand/pi05_h2_dianzha_down_no_hand/19999 \
  --sample-indices 200 2500 7500 12000 17000 \
  --output-dir visualization_output/h2_dianzha_down_no_hand
```


```bash
HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions.py \
  --config-name pi05_h2_one_hand_mixu \
  --checkpoint-dir ../../weights/openpi/checkpoints/pi05_h2_one_hand_mixu/pi05_h2_one_hand/19999 \
  --sample-indices 200 2500 7500 12000 17000 \
  --output-dir visualization_output/pi05_h2_one_hand_mixu
```

为获得更有代表性的结果，可从 20 个均匀分布的 episode 中各取一个中间帧进行评估。该模式会避开
episode 首尾，并在 `summary.json` 和 `per_sample_metrics.csv` 中记录 episode 索引；不能与
`--sample-indices` 同时使用：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions.py \
  --config-name pi05_h2_one_hand_mixu \
  --checkpoint-dir ../../weights/openpi/checkpoints/pi05_h2_one_hand_mixu/pi05_h2_one_hand/19999 \
  --num-episodes 20 \
  --output-dir visualization_output/pi05_h2_one_hand_mixu_20_episodes
```

### 全训练数据评估与高误差分析

使用 `--all-samples` 会逐帧评估整个训练数据集。该过程需要读取每一帧的三路视频并执行一次模型推理，
耗时明显长于抽样评估。为避免生成极大的文件，此模式不会保存全部动作块或每帧轨迹图，而会生成：

- `per_sample_metrics.csv`：每个训练帧的误差；
- `per_episode_metrics.csv`：按 episode 聚合的 MAE、RMSE、首动作误差和最大误差；
- `high_error_samples.csv`：按 `first_action_mae`（再按 `mae`）排序的前 20 个高误差样本；
- `error_by_episode.png`：每个 episode 的全动作块 MAE 与首动作 MAE，红色点表示首动作误差最大的 10 个 episode；
- 仍会生成全数据的关节误差图和时间步误差热图。

```bash
HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions.py \
  --config-name pi05_h2_one_hand_mixu \
  --checkpoint-dir ../../weights/openpi/checkpoints/pi05_h2_one_hand_mixu/pi05_h2_one_hand/19999 \
  --all-samples \
  --top-k 20 \
  --output-dir visualization_output/pi05_h2_one_hand_mixu_all_samples
```

指定双手 checkpoint、样本和独立输出目录，例如比较 step 19999：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions.py \
  --config-name pi05_h2_two_hand \
  --checkpoint-dir ../../weights/openpi/checkpoints/pi05_h2_two_hand/two_hand_v1/19999 \
  --sample-indices 200 2500 7500 12000 17000 \
  --output-dir visualization_output/h2_checkpoint_19999
```

### 直接查看训练数据推理输出

若只想查看模型在记录训练数据上的输入和输出，不生成图片或结果文件，可运行：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/infer_h2_dataset.py
```

脚本默认读取单手最终 checkpoint `../../weights/openpi/checkpoints/pi05_h2_one_hand/smoke_test/19999`，并推理样本
`0 1000 5000 10000 15000`。终端会打印每个样本的任务文本、三路图像 shape、29 维 state、原始
`(50, 32)` 模型输出、前 29 维 H2 预测动作、记录标签、逐元素绝对误差和汇总指标。原始输出最后
3 维是 padding，不是 H2 关节。

先快速查看单个样本时，可指定索引：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/infer_h2_dataset.py \
  --sample-indices 0
```

### 服务端推理客户端测试

可将模型作为 WebSocket 服务运行，使相机和机器人状态采集程序与模型推理进程分离。先启动服务端：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/serve_policy.py \
  --port 9000 \
  policy:checkpoint \
  --policy.config=pi05_h2_one_hand \
  --policy.dir=../../weights/openpi/checkpoints/pi05_h2_one_hand/smoke_test/19999
```

在另一个终端使用记录训练数据测试客户端。客户端会发送三路图像、29 维 state 和任务文本，打印服务端
返回的 `(50, 32)` 动作、前 29 维 H2 关节动作、padding、时延、记录标签和误差；不会向机器人发送命令：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/client_h2_dataset.py \
  --host 127.0.0.1 \
  --port 9000 \
  --sample-indices 0
```

`client_h2_dataset.py` 末尾已说明真机接入边界：把训练数据的三路 `uint8` HWC 图像和 29 维 state
替换为实时采集数据，同时保持 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 的键名和 state 维度顺序不变。
返回的 50 步动作块不能直接执行；真机控制端必须先施加关节限位、速度限制和单周期增量限制，并在每个闭环周期只执行 1 至 3 步。

## 7. 如何解读离线指标

`summary.json` 包含以下指标：

| 指标 | 含义 | 优先级 |
| --- | --- | --- |
| `mae` | 全部样本、50 步、29 维动作的平均绝对误差 | 中 |
| `rmse` | 对较大误差更敏感的均方根误差 | 中 |
| `max_abs_error` | 所有动作值中的最大单点绝对误差 | 用于排查 |
| `first_action_mae` | 每个动作块第 0 步的平均绝对误差 | 高 |
| `first_action_rmse` | 每个动作块第 0 步的 RMSE | 高 |
| `mean_infer_ms` | 平均模型采样时间 | 高，影响控制频率 |

对于 15 Hz 闭环控制，一个控制周期预算约为 $1000 / 15 = 66.7$ ms。实际还需扣除图像采集、网络传输、状态读取和控制下发时间，因此模型 `mean_infer_ms` 越低越好。

当前一次 `19999` checkpoint 离线检查的结果为：

```json
{
  "mae": 0.04519335553050041,
  "rmse": 0.08163107931613922,
  "max_abs_error": 0.5065728425979614,
  "first_action_mae": 0.03159449249505997,
  "first_action_rmse": 0.055943313986063004
}
```

换算为角度：

- 全动作块 MAE 约为 2.59 度。
- 第一个动作 MAE 约为 1.81 度。
- 最大单点误差约为 29.0 度。

对于闭环 receding-horizon 控制，应优先关注 `first_action_mae` 和第一个动作是否安全、平滑。50 步动作块覆盖约 3.33 秒，块后半段误差通常会比第一个动作大；不要把整段 50 步动作开环执行。

## 8. 输出文件与图片

每次离线检查会保存：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | checkpoint、样本索引、汇总指标、平均推理耗时和图片路径 |
| `per_sample_metrics.csv` | 每个抽样帧的指标和任务文本 |
| `action_chunks.npz` | `predictions` 与 `targets`，shape 均为 `(样本数, 50, 29)` |
| `mean_absolute_error_heatmap.png` | 时间步 x 关节维度的平均绝对误差热图 |
| `mean_absolute_error_by_joint.png` | 每个关节维度的平均绝对误差柱状图 |
| `sample_XXXXX_action_comparison.png` | 一个样本的目标/预测轨迹和绝对误差图 |

查看图片时按以下顺序排查：

1. 看 `mean_absolute_error_by_joint.png`：误差是否集中在手指 `[17:29]`，还是腰/双臂 `[0:17]`。
2. 看 `mean_absolute_error_heatmap.png`：高误差是否只出现在动作块后段，还是第 0 到第 2 步已明显偏大。
3. 打开对应 `sample_XXXXX_action_comparison.png`：检查高误差关节是否存在突然跳变、预测震荡或超出正常关节角范围。
4. 对照 `per_sample_metrics.csv` 的 `sample_index` 和 `prompt`，定位具体轨迹和任务。

## 9. 比较不同 checkpoint

建议比较 `5000`、`10000`、`15000` 和 `19999`，每个 checkpoint 使用完全相同的样本索引。这样结果可横向比较：

```bash
for step in 5000 10000 15000 19999; do
  HF_HUB_OFFLINE=1 uv run python scripts/check_h2_offline_predictions.py \
    --checkpoint-dir "../../weights/openpi/checkpoints/pi05_h2_one_hand/smoke_test/${step}" \
    --sample-indices 0 1000 5000 10000 15000 \
    --output-dir "visualization_output/h2_checkpoint_${step}"
done
```

优先选择满足以下条件的 checkpoint 进入真实机器人保守测试：

- `first_action_mae` 较低。
- 腰和双臂的第 0 到第 2 步没有尖峰误差。
- 预测动作轨迹平滑，没有明显高频抖动。
- `mean_infer_ms` 满足实际控制链路的时延预算。

仅在训练样本上离线误差低，不能证明真实机器人泛化。最好保留若干未参与训练的 episode 作为独立离线验证集，并在固定初始条件和扰动初始条件下分别测试。

## 10. 上机前安全清单

离线验证通过后，仍应按以下顺序上机：

1. 先只记录预测动作，不下发给机器人。
2. 确认所有输出均为有限值，没有 `NaN` 或 `inf`。
3. 对 29 维输出应用关节位置、速度和单周期最大增量限制。
4. 首先在悬空、无接触或安全场景测试。
5. 以 receding horizon 方式每次只执行 1 到 3 步，然后重新获取图像和 state 并推理。
6. 不要开环执行完整的 50 步动作块。
7. 双手输出 `[17:29]` 的 12 个组均值维度时，按机器人控制接口要求展开到实际手指关节，并保留限位和急停机制。

训练和离线检查都只能说明模型对记录数据的拟合程度。真实任务成功率、碰撞风险、动作平滑性和对初始状态扰动的鲁棒性，必须通过受控的闭环机器人测试确认。
