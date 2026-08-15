# H2 pi0.5 MixU WebSocket 服务端与客户端推理指南

本文档说明如何将已训练完成的 H2 单手模型作为 WebSocket 服务运行，并通过本地训练数据客户端验证服务端推理。所有命令均在仓库根目录执行。

> **安全边界**：默认情况下，`scripts/client_h2_dataset.py` 只发送记录数据，并只打印灵巧手寄存器换算结果。加入 `--send-hands` 后，它会把预测手部动作直接写入串口，从而实际驱动灵巧手；腰和双臂动作仍不会由此脚本下发。真机控制必须具备急停、人工监控、关节位置/速度/单周期增量限制，且不得直接执行模型返回的完整 50 步动作块。

## 1. 模型与端口

| 项目 | 值 |
| --- | --- |
| 训练配置 | `pi05_h2_one_hand_mixu` |
| checkpoint | `../../weights/openpi/checkpoints/pi05_h2_one_hand_mixu/pi05_h2_one_hand/19999` |
| 本地数据集 | `data_train/lerobot_h2_one_hand_15hz_8_10_mixu` |
| 默认端口 | `9000` |
| 输入 state | 29 维 `float32` 绝对关节角，单位 rad |
| 输出动作 | `(50, 32)`，前 29 维为 H2 关节动作，后 3 维为 padding |

开始前确认 checkpoint 目录内至少存在 `assets`、`params` 与 `train_state`：

```bash
find ../../weights/openpi/checkpoints/pi05_h2_one_hand_mixu/pi05_h2_one_hand/19999 \
  -maxdepth 1 -type d | sort
```

若已下载模型与依赖，可启用离线模式：

```bash
export HF_HUB_OFFLINE=1
```

## 2. 启动模型服务端

在**终端 1**运行：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/serve_policy.py \
  --port 9000 \
  policy:checkpoint \
  --policy.config=h2_dianzha_down_no_hand \
  --policy.dir=../../weights/openpi/checkpoints/h2_dianzha_down_no_hand/pi05_h2_dianzha_down_no_hand/19999
```



```bash
HF_HUB_OFFLINE=1 uv run python scripts/serve_policy.py \
  --port 9000 \
  policy:checkpoint \
  --policy.config=pi05_h2_one_hand_mixu \
  --policy.dir=../../weights/openpi/checkpoints/pi05_h2_one_hand_mixu/pi05_h2_one_hand/19999
```



服务端会加载 checkpoint，并监听 `0.0.0.0:9000`。首次加载会花费较长时间；在日志显示服务已创建后保持此终端运行。

若只在本机测试，客户端连接地址为 `127.0.0.1`。停止服务时，在服务端终端按 `Ctrl+C`。

### 可选：记录服务端策略输入输出

调试时可在服务端命令中加入 `--record`：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/serve_policy.py \
  --port 9000 \
  --record \
  policy:checkpoint \
  --policy.config=pi05_h2_one_hand_mixu \
  --policy.dir=../../weights/openpi/checkpoints/pi05_h2_one_hand_mixu/pi05_h2_one_hand/19999
```

这会在仓库根目录写入策略记录，仅用于调试；正常运行不需要此选项。

## 3. 使用记录数据测试客户端

保持服务端运行，在**终端 2**执行单样本验证：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/client_h2_dataset.py \
  --host 127.0.0.1 \
  --port 9000 \
  --config-name pi05_h2_one_hand_mixu \
  --sample-indices 145
```

客户端会执行以下操作：

1. 从 MixU 本地数据集读取三路图像、29 维 state 和任务文本。
2. 通过 WebSocket 发送观测数据到服务端。
3. 打印服务端元数据、首个动作的关节名称和值、时延和与记录动作的误差。
4. 仅在本地比较预测与标签，不会控制机器人，也不会写入评估图片或结果文件。

可使用多个样本复查不同 episode；以下样本来自 20 episode 均匀评估：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/client_h2_dataset.py \
  --host 127.0.0.1 \
  --port 9000 \
  --config-name pi05_h2_one_hand_mixu \
  --sample-indices 145 8137 48491 62099
```

注意：客户端只打印第 0 步的 29 个关节动作及手部寄存器换算结果。一次测试多个样本时，若加入
`--send-hands`，每个样本都会触发一次左右手串口写入，因此实际控制时必须只指定一个样本。

### 连续 100 次推理时延测试

`scripts/benchmark_h2_policy_client.py` 会加载一次记录样本，然后向服务端连续发送同一份观测。它不会向
机器人或串口发送任何动作，适合独立测量模型、服务端和 WebSocket 往返时延。默认先执行 5 次预热，再统计
100 次请求的均值、标准差、P50、P95、P99、最小值、最大值与客户端观测吞吐量：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/benchmark_h2_policy_client.py \
  --host 127.0.0.1 \
  --port 9000 \
  --config-name pi05_h2_one_hand_mixu \
  --sample-index 145 \
  --warmup-requests 5 \
  --requests 100
```

输出包括：

- `Policy inference`：策略进程内模型采样时间；
- `Server inference`：服务端处理一次请求的时间；
- `Client round trip`：客户端从请求发送至收到响应的总时间；
- `Estimated transport + serialization`：`round_trip - server_inference`，其中包括 WebSocket 序列化、调度与网络传输，**不是**纯网络 ping 延迟。

如需测试远程服务器，将 `--host` 替换为服务端 IP。该脚本只加载一次图像和 state，因此结果不包含真机相机采集、实时 state 读取或控制下发时延。

### 灵巧手 Modbus 串口换算与控制

客户端把已反归一化的手部动作 `dims 17–28` 映射为两只手各 6 个 `angleSet` 寄存器值：

| 模型维度 | 手部 | 6 个自由度顺序 |
| --- | --- | --- |
| `17:23` | 左手 | 小指、无名指、中指、食指、拇指弯曲、拇指侧摆 |
| `23:29` | 右手 | 小指、无名指、中指、食指、拇指弯曲、拇指侧摆 |

模型的每个手指值是组均值（rad），不是直接的 Modbus 寄存器值。客户端的
`hand_rad6_to_angles()` 先将组均值转换为组内关节弧度和，再映射到寄存器量程：

$$
s_i = n_i r_i, \qquad
u_i = \operatorname{clip}\!\left(\frac{s_i-s_{\min}}{s_{\max}-s_{\min}}, 0, 1\right), \qquad
\operatorname{angleSet}_i = \left\lfloor o_{\max} - u_i(o_{\max}-o_{\min}) \right\rfloor
$$

其中四指的 $n_i=2$、拇指弯曲 $n_i=3$、拇指侧摆 $n_i=1$。换算会限制在寄存器量程内，默认只打印下列对照信息，不会写串口：组内弧度和、左手/右手 `angleSet` 值。

#### 先进行干运行

```bash
HF_HUB_OFFLINE=1 uv run python scripts/client_h2_dataset.py \
  --host 127.0.0.1 \
  --port 9000 \
  --config-name pi05_h2_one_hand_mixu \
  --sample-indices 145
```

确认打印的左右手 `angleSet` 值、接线端口和机械初始姿态后，才可考虑串口下发。

打印数据示例：
```css
(base) liguopu@untu-System-Product-Name:~/lgp_dev/project/openpi$ HF_HUB_OFFLINE=1 uv run python scripts/client_h2_dataset.py   --host 127.0.0.1   --port 9000   --config-name pi05_h2_one_hand_mixu   --sample-indices 145
/home/liguopu/lgp_dev/project/openpi/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
2026-08-11 17:02:31.975359: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-08-11 17:02:31.975394: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-08-11 17:02:31.976040: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-08-11 17:02:32.361633: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Connecting to H2 policy server: ws://127.0.0.1:9000
Dataset: data_train/lerobot_h2_one_hand_15hz_8_10_mixu (62294 samples)
Requested samples: [145]
Hand serial in dry-run mode: converted register values are printed only.
Safety: arms/waist actions are never commanded by this client.

Server metadata: {}

/home/liguopu/lgp_dev/project/openpi/.venv/lib/python3.11/site-packages/torchvision/io/_video_deprecation_warning.py:5: UserWarning: The video decoding and encoding capabilities of torchvision are deprecated from version 0.22 and will be removed in version 0.24. We recommend that you migrate to TorchCodec, where we'll consolidate the future decoding/encoding capabilities of PyTorch: https://github.com/pytorch/torchcodec
  warnings.warn(
====================================================================================================
Dataset sample 145
Input prompt: He picked up the green and black voltage testers and placed them aside
Input images:
  cam_high: shape=(480, 640, 3), dtype=uint8, range=[0, 255]
  cam_left_wrist: shape=(480, 640, 3), dtype=uint8, range=[0, 255]
  cam_right_wrist: shape=(480, 640, 3), dtype=uint8, range=[0, 255]
Input state (29 H2 joint positions, rad) shape=(29,), dtype=float32
[ 0.00049 -0.02826 -0.15353  0.42874  0.19667 -0.08821  0.02952  0.01404 -0.00413  0.12982 -0.59346 -0.32739 -0.10565  0.66658 -0.10601 -0.30706  0.05699
  0.54303  0.48472  0.52908  0.58005  0.1352   0.4753   1.27402  0.99035  0.74726  0.72524  0.1412   1.30706]
Response keys: ['actions', 'policy_timing', 'server_timing', 'state']
  client_round_trip_ms: 112.1 ms, including serialization and network transfer.
Raw server actions shape=(50, 32), dtype=float32
  (50 steps, 32 dimensions) means 50 predicted future actions at 15 Hz (about 3.33 s).
  Dimensions 0-28 are 29 absolute H2 joint-position targets in rad;
  dimensions 29-31 are model padding and must not be sent to the robot.
  Showing only step 0, the earliest action for closed-loop control:
    [ 0] waist_yaw_joint             -0.00019 rad
    [ 1] waist_roll_joint            -0.02555 rad
    [ 2] waist_pitch_joint           -0.14482 rad
    [ 3] left_shoulder_pitch_joint    0.42533 rad
    [ 4] left_shoulder_roll_joint     0.21538 rad
    [ 5] left_shoulder_yaw_joint     -0.09621 rad
    [ 6] left_elbow_joint             0.03898 rad
    [ 7] left_wrist_roll_joint        0.02943 rad
    [ 8] left_wrist_pitch_joint       0.00382 rad
    [ 9] left_wrist_yaw_joint         0.14675 rad
    [10] right_shoulder_pitch_joint  -0.61958 rad
    [11] right_shoulder_roll_joint   -0.47594 rad
    [12] right_shoulder_yaw_joint    -0.05053 rad
    [13] right_elbow_joint            0.74377 rad
    [14] right_wrist_roll_joint      -0.53020 rad
    [15] right_wrist_pitch_joint     -0.22972 rad
    [16] right_wrist_yaw_joint       -0.44771 rad
    [17] left_little                  0.52640 rad
    [18] left_ring                    0.50815 rad
    [19] left_middle                  0.53342 rad
    [20] left_index                   0.57337 rad
    [21] left_thumb_bend              0.12657 rad
    [22] left_thumb_side              0.51554 rad
    [23] right_little                 1.16089 rad
    [24] right_ring                   1.02058 rad
    [25] right_middle                 0.81954 rad
    [26] right_index                  0.72048 rad
    [27] right_thumb_bend             0.15068 rad
    [28] right_thumb_side             1.22346 rad
    [29:32] padding (ignore)           [-0.02642822 -0.00665975  0.00483096]
Hand serial conversion (action step 0, dims 17-28):
  DOF           left rad(sum)  left reg    right rad(sum)  right reg
  little               1.0528      1479            2.3218       1166
  ring                 1.0163      1488            2.0412       1235
  middle               1.0668      1476            1.6391       1335
  index                1.1467      1456            1.4410       1383
  thumb_bend           0.3797      1290            0.4520       1279
  thumb_side           0.5155      1474            1.2235       1027
  left  angleSet -> [1479, 1488, 1476, 1456, 1290, 1474]
  right angleSet -> [1166, 1235, 1335, 1383, 1279, 1027]
  (dry-run: pass --send-hands to actually write the serial ports)
Policy timing: {'infer_ms': 12.038945918902755}
Server timing: {'infer_ms': 101.80017305538058}
Comparison metrics: {'mae': 0.07121215015649796, 'first_action_mae': 0.0729699432849884, 'max_abs_error': 0.7474187612533569}
====================================================================================================
Aggregate recorded-data metrics: {'mae': 0.07121215015649796, 'first_action_mae': 0.0729699432849884, 'max_abs_error': 0.7474187612533569}

For real-robot use, replace _make_h2_observation_from_dataset with live uint8 HWC images
and a live float32 29-value joint state in exactly the same camera keys and state ordering.
Do not directly execute this 50-step output block: apply limits and execute only 1-3 steps per control cycle.
(base) liguopu@untu-System-Product-Name:~/lgp_dev/project/openpi$ 
```

#### 显式允许写入串口

串口功能使用 Python `pyserial`。当前项目环境未默认安装该可选依赖，首次使用前需在仓库环境中安装它。

```bash
uv add pyserial
```

默认左手端口为 `/dev/ttyUSB3`，右手端口为 `/dev/ttyUSB2`，Modbus slave ID 为 `1`，波特率为 `115200`，写入起始寄存器为 `1040`。如设备枚举顺序不同，必须通过 `--left-port` 和 `--right-port` 显式指定正确端口。

以下命令会对单个记录样本的第 0 步预测动作进行换算，并直接写入两只灵巧手：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/client_h2_dataset.py \
  --host 127.0.0.1 \
  --port 9000 \
  --config-name pi05_h2_one_hand_mixu \
  --sample-indices 145 \
  --hand-step 0 \
  --left-port /dev/ttyUSB3 \
  --right-port /dev/ttyUSB2 \
  --send-hands
```

> `--send-hands` 没有速度限制、反馈闭环或急停逻辑；目前它仅构造 Modbus RTU CRC16 帧并以功能码 `0x10` 写入 6 个 `angleSet` 寄存器。仅可在手部悬空、无接触、急停有效且有人监控的条件下使用。先以干运行核对数值，再用单样本进行低风险验证；不得用默认的多个样本连续下发。

## 4. 如何阅读客户端输出

客户端应返回以下关键字段：

| 字段 | 含义 | 使用要求 |
| --- | --- | --- |
| `actions` | 模型动作块，shape 为 `(50, 32)` | 真机仅可考虑前 29 维，且每次闭环最多执行 1 至 3 步 |
| `state` | 策略变换返回的诊断 state | 不是机器人命令 |
| `policy_timing.infer_ms` | 策略进程内模型采样耗时 | 用于估计纯模型时延 |
| `server_timing.infer_ms` | 服务端总推理耗时 | 包含服务端处理开销 |
| `client_round_trip_ms` | 客户端往返耗时 | 包含序列化、网络和服务端耗时 |
| `Comparison metrics` | 与记录标签的离线误差 | 仅用于评估，不是安全保证 |

### 归一化与反归一化

**不需要在客户端或机器人控制端再次进行反归一化。** 服务端从 checkpoint 的 `assets` 中加载训练时保存的
归一化统计，并自动完成以下流程：

1. 客户端发送的 `state` 是原始物理关节角（rad）；服务端在模型输入前自动归一化。
2. 模型在归一化动作空间中采样。
3. 服务端在发送 WebSocket 响应前，自动把前 29 维动作反归一化为训练数据的原始物理单位和语义：绝对关节角（rad）。

因此，客户端打印的 `actions[:, :29]` 已经是反归一化后的绝对目标关节角。再次按 mean/std 或 quantile
统计进行反归一化会产生错误的控制值。`actions[:, 29:32]` 不属于 H2 关节，也没有控制语义，必须忽略。

这不等同于“可直接下发”：机器人控制端仍需根据实际 H2 控制接口执行关节名称/顺序映射、手指组维度展开、
位置限位、速度限位和单周期增量限制。

第一条请求通常包含模型预热，耗时明显更高。应使用后续请求的 `policy_timing.infer_ms` 评估稳定推理性能。

本模型的 20 episode 均匀离线评估结果为：全动作块 MAE $0.0551$ rad（约 $3.16^\circ$），首动作 MAE $0.0407$ rad（约 $2.33^\circ$），最大单点误差 $0.9623$ rad（约 $55.1^\circ$）。因此，离线测试成功不表示动作可直接下发至真机。



## 5. 远程客户端连接

服务端绑定所有网卡，可信内网中的另一台机器可连接服务器 IP。将 `<SERVER_IP>` 替换为服务端机器的实际地址：

```bash
HF_HUB_OFFLINE=1 uv run python scripts/client_h2_dataset.py \
  --host <SERVER_IP> \
  --port 9000 \
  --config-name pi05_h2_one_hand_mixu \
  --sample-indices 145
```

远程运行此记录数据客户端时，客户端机器也必须具备本仓库环境和 MixU 数据集，因为脚本会在本地读取记录样本。若只需真机实时推理，应由机器人侧程序按照下一节的输入协议实现客户端，而不是依赖本脚本。

> 该 WebSocket 服务不提供认证或传输加密。只能部署在受信任的隔离网络中；不要将端口 `9000` 暴露到公网。必要时在防火墙中仅允许机器人客户端的 IP 访问 TCP `9000`。

## 6. 真机客户端输入输出协议

机器人侧每个闭环周期需要发送一个 observation 字典，字段名、数组类型和维度必须与训练保持一致：

```text
images.cam_high:        uint8 HWC RGB 图像，头部相机
images.cam_left_wrist:  uint8 HWC RGB 图像，左腕相机
images.cam_right_wrist: uint8 HWC RGB 图像，右腕相机
state:                  float32，shape=(29,)，绝对关节角，单位 rad
prompt:                 str，任务文本
```

服务端返回的 `actions` 为 `(50, 32)`。真机侧必须：

1. 检查所有值均为有限数，拒绝 `NaN` 与 `inf`。
2. 只取已反归一化的 `actions[:, :29]`；最后 3 维仅是模型 padding。
3. 按机器人控制接口确认 29 维关节的顺序与训练数据完全一致，并按控制接口要求展开 12 个手指组维度。
4. 对位置、速度与单周期增量应用硬限位。
5. 初始阶段只执行第 0 步；确认安全后，闭环中每周期最多执行 1 至 3 步。
6. 每次执行后重新采集图像和 state 并重新请求模型，不得开环执行完整 50 步。
7. 保持急停、人工监控和无接触/悬空等保守测试条件。

## 7. 常见问题

### `Connection refused`

确认服务端终端仍在运行，且客户端 `--host`、`--port` 与服务端一致。远程连接还需检查两台机器的网络可达性和防火墙规则。

### `Checkpoint directory does not exist` 或模型加载失败

确认服务端命令中的 `--policy.dir` 精确指向 `19999` 目录，并检查该目录下存在 `assets`、`params` 和 `train_state`。

### 客户端报告图像键、state 维度或动作 shape 不匹配

实时客户端必须使用 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三个键，并按照训练数据的 29 维 state 关节顺序构造 `float32` 数组。服务端返回的正确动作 shape 是 `(50, 32)`。

### 首次请求耗时很长

这是模型加载/编译或 GPU 预热的正常现象。先发送一次不执行的预热请求，再以之后请求的稳定时延评估控制周期。

### `No module named 'serial'`

这是缺少灵巧手串口依赖 `pyserial`。安装该依赖后重新运行客户端；不使用 `--send-hands` 的干运行模式不会导入串口库。

### 出现 `torchvision` 视频解码弃用警告

这是记录数据客户端读取 LeRobot 视频时的依赖警告，不影响当前离线客户端测试。它与 WebSocket 模型输出无关。
