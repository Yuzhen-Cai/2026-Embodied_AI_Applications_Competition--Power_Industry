# ruff: noqa: RUF002, RUF003
"""Test an H2 WebSocket policy server with recorded local training samples.

新增：灵巧手 dims 17-28 直接驱动串口（绕过 ROS），并打印直接控制值。

默认只打印换算结果，不写串口；加 --send-hands 才真正驱动：
    python client_h2_dataset.py --sample-indices 145                 # 只看换算值
    python client_h2_dataset.py --sample-indices 145 --send-hands    # 真发串口
"""

import time

import numpy as np
from openpi_client import websocket_client_policy
import tyro

from openpi.training import config as training_config
from openpi.training import data_loader

H2_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_little",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb_bend",
    "left_thumb_side",
    "right_little",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb_bend",
    "right_thumb_side",
)

# ============== 灵巧手配置（与 hand_teleop.py 完全一致）==============
# 每手 6 个自由度: [小指, 无名指, 中指, 食指, 拇指弯曲, 拇指侧摆]
# (组内关节数, (寄存器out_min, out_max), (弧度in_min, in_max))
# in 范围 = 组内关节弧度求和的范围；模型 6 维是组均值，求和 = 均值 × 组内关节数
FINGER_MAP = [
    (2, (900, 1740), (0.0, 3.4)),    # 小指   (×2 关节)
    (2, (900, 1740), (0.0, 3.4)),    # 无名指 (×2 关节)
    (2, (900, 1740), (0.0, 3.4)),    # 中指   (×2 关节)
    (2, (900, 1740), (0.0, 3.4)),    # 食指   (×2 关节)
    (3, (1100, 1350), (0.0, 1.6)),   # 拇指弯 (×3 关节)
    (1, (600, 1800), (0.0, 1.9)),    # 拇指侧摆 (×1 关节)
]
HAND_DOF_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_side"]
LEFT_HAND_SLICE = slice(17, 23)   # 模型 dims 17-22 左手
RIGHT_HAND_SLICE = slice(23, 29)  # 模型 dims 23-28 右手

# 串口/Modbus 参数（与 hand_teleop.py 一致）
DEFAULT_LEFT_PORT = "/dev/ttyUSB3"
DEFAULT_RIGHT_PORT = "/dev/ttyUSB2"
DEFAULT_SLAVE_ID = 1
DEFAULT_BAUDRATE = 115200
ANGLE_SET_START = 1040
OPEN_TARGET = [1740, 1740, 1740, 1740, 1350, 1800]


# ============== Modbus 协议（复刻 hand_teleop.py 的实现）==============
def crc16_modbus(payload: bytes) -> bytes:
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def build_write_multiple_registers_request(slave_id: int, address: int, values: list[int]) -> bytes:
    data = []
    for value in values:
        data.extend([value >> 8, value & 0xFF])
    payload = bytes([
        slave_id, 0x10,
        address >> 8, address & 0xFF,
        0, len(values), len(data), *data,
    ])
    return payload + crc16_modbus(payload)


class HandSerialClient:
    """单只灵巧手的串口客户端（写 angleSet 寄存器）。"""

    def __init__(self, port: str, slave_id: int = DEFAULT_SLAVE_ID,
                 baudrate: int = DEFAULT_BAUDRATE, timeout: float = 0.5):
        self.port = port
        self.slave_id = slave_id
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_port = None

    def open(self) -> None:
        if self.serial_port is None:
            import serial  # noqa: PLC0415
            self.serial_port = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self) -> None:
        if self.serial_port is not None:
            self.serial_port.close()
            self.serial_port = None

    def send_angles(self, values: list[int]) -> None:
        self.open()
        request = build_write_multiple_registers_request(self.slave_id, ANGLE_SET_START, values)
        written = self.serial_port.write(request)
        if written != len(request):
            raise OSError(f"{self.port}: expected {len(request)} bytes, wrote {written}")
        self.serial_port.flush()


# ============== 弧度 → 寄存器角度换算（hand_teleop.py 映射的等价实现）==============
def hand_rad6_to_angles(rad6: np.ndarray) -> tuple[list[int], list[float]]:
    """模型输出的一手 6 维组均值(rad) → 6 个 Modbus angleSet 寄存器值。

    返回 (寄存器值列表, 组内求和弧度列表)，后者仅用于打印核对。
    """
    angles, raw_sums = [], []
    for value, (n_joints, (out_min, out_max), (in_min, in_max)) in zip(rad6, FINGER_MAP, strict=True):
        raw = float(value) * n_joints          # 组均值 × 关节数 = 组内求和（expand_hand 的效果）
        ratio = np.clip((raw - in_min) / (in_max - in_min), 0.0, 1.0)
        angles.append(int(out_max - ratio * (out_max - out_min)))
        raw_sums.append(raw)
    return angles, raw_sums


def drive_hands_from_action(action29: np.ndarray, send: bool,  # noqa: FBT001
                            left_port: str = DEFAULT_LEFT_PORT,
                            right_port: str = DEFAULT_RIGHT_PORT,
                            step: int = 0) -> None:
    """取动作块第 step 步的 dims 17-28，换算成串口控制值，打印并（可选）直发串口。"""
    left_angles, left_raw = hand_rad6_to_angles(action29[LEFT_HAND_SLICE])
    right_angles, right_raw = hand_rad6_to_angles(action29[RIGHT_HAND_SLICE])

    print(f"Hand serial conversion (action step {step}, dims 17-28):")
    print(f"  {'DOF':<12} {'left rad(sum)':>14} {'left reg':>9}   "
          f"{'right rad(sum)':>15} {'right reg':>10}")
    for i, name in enumerate(HAND_DOF_NAMES):
        print(f"  {name:<12} {left_raw[i]:>14.4f} {left_angles[i]:>9}   "
              f"{right_raw[i]:>15.4f} {right_angles[i]:>10}")
    print(f"  left  angleSet -> {left_angles}")
    print(f"  right angleSet -> {right_angles}")

    if not send:
        print("  (dry-run: pass --send-hands to actually write the serial ports)")
        return

    clients = []
    try:
        for port, angles, side in ((left_port, left_angles, "left"),
                                   (right_port, right_angles, "right")):
            client = HandSerialClient(port)
            client.open()
            client.send_angles(angles)
            clients.append(client)
            print(f"  SENT {side:<5} {port}: angleSet={angles}")
    finally:
        for client in clients:
            client.close()


# ============== 以下为原有客户端逻辑 ==============
def _make_h2_observation_from_dataset(sample: dict) -> dict:
    """Map one recorded LeRobot H2 sample to the server observation contract."""
    return {
        "images": {
            "cam_high": sample["observation.images.head"],
            "cam_left_wrist": sample["observation.images.left"],
            "cam_right_wrist": sample["observation.images.right"],
        },
        "state": sample["observation.state"],
        "prompt": sample["task"],
    }


def _print_array(name: str, values: np.ndarray) -> None:
    """Print all values of a small diagnostic array."""
    print(f"{name} shape={values.shape}, dtype={values.dtype}")
    print(np.array2string(values, precision=5, suppress_small=True, threshold=np.inf, max_line_width=160))


def _print_first_action(raw_actions: np.ndarray) -> None:
    """Explain the action chunk and print only its first executable H2 action."""
    print(f"Raw server actions shape={raw_actions.shape}, dtype={raw_actions.dtype}")
    print("  (50 steps, 32 dimensions) means 50 predicted future actions at 15 Hz (about 3.33 s).")
    print("  Dimensions 0-28 are 29 absolute H2 joint-position targets in rad;")
    print("  dimensions 29-31 are model padding and must not be sent to the robot.")
    print("  Showing only step 0, the earliest action for closed-loop control:")
    for dimension, (joint_name, value) in enumerate(zip(H2_JOINT_NAMES, raw_actions[0, :29], strict=True)):
        print(f"    [{dimension:2d}] {joint_name:<27} {value: .5f} rad")
    print(f"    [29:32] padding (ignore)           {raw_actions[0, 29:32]}")


def _print_image_summary(images: dict[str, np.ndarray]) -> None:
    """Print image input metadata without flooding the terminal with pixels."""
    for camera_name, image in images.items():
        image_array = np.asarray(image)
        print(
            f"  {camera_name}: shape={image_array.shape}, dtype={image_array.dtype}, "
            f"range=[{image_array.min():.0f}, {image_array.max():.0f}]"
        )


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Calculate comparison metrics for one 50-step, 29-joint action chunk."""
    errors = prediction - target
    first_action_errors = errors[0] if errors.ndim == 2 else errors[:, 0, :]
    return {
        "mae": float(np.mean(np.abs(errors))),
        "first_action_mae": float(np.mean(np.abs(first_action_errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
    }


def main(
    host: str = "127.0.0.1",
    port: int = 9000,
    config_name: str = "pi05_h2_one_hand",
    sample_indices: tuple[int, ...] = (0, 1000, 5000, 10000, 15000),
    send_hands: bool = False,  # noqa: FBT001, FBT002
    hand_step: int = 0,
    left_port: str = DEFAULT_LEFT_PORT,
    right_port: str = DEFAULT_RIGHT_PORT,
) -> None:
    """Send recorded H2 observations to a policy server and print its responses.

    send_hands: actually write dims 17-28 to the dexterous-hand serial ports
                (default False = print the converted register values only).
    hand_step:  which step of the 50-step action chunk to use for the hands.
    """
    if not sample_indices:
        raise ValueError("At least one sample index is required.")

    config = training_config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    invalid_indices = [index for index in sample_indices if index < 0 or index >= len(dataset)]
    if invalid_indices:
        raise IndexError(f"Sample indices outside dataset range [0, {len(dataset) - 1}]: {invalid_indices}")

    print(f"Connecting to H2 policy server: ws://{host}:{port}")
    print(f"Dataset: {data_config.data_root} ({len(dataset)} samples)")
    print(f"Requested samples: {list(sample_indices)}")
    if send_hands:
        print(f"HAND SERIAL OUTPUT ENABLED: left={left_port}, right={right_port}, step={hand_step}")
    else:
        print("Hand serial in dry-run mode: converted register values are printed only.")
    print("Safety: arms/waist actions are never commanded by this client.\n")

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"Server metadata: {client.get_server_metadata()}\n")
    all_predictions, all_targets = [], []

    for index in sample_indices:
        sample = dataset[index]
        observation = _make_h2_observation_from_dataset(sample)
        print("=" * 100)
        print(f"Dataset sample {index}")
        print(f"Input prompt: {observation['prompt']}")
        print("Input images:")
        _print_image_summary(observation["images"])
        _print_array("Input state (29 H2 joint positions, rad)", np.asarray(observation["state"]))

        start_time = time.perf_counter()
        response = client.infer(observation)
        client_round_trip_ms = (time.perf_counter() - start_time) * 1000
        print(f"Response keys: {sorted(response)}")
        print(f"  client_round_trip_ms: {client_round_trip_ms:.1f} ms, including serialization and network transfer.")

        raw_actions = np.asarray(response["actions"], dtype=np.float32)
        expected_shape = (config.model.action_horizon, config.model.action_dim)
        if raw_actions.shape != expected_shape:
            raise ValueError(f"Unexpected server action shape {raw_actions.shape}; expected {expected_shape}.")
        prediction = raw_actions[:, :29]
        target = np.asarray(sample["action"], dtype=np.float32)

        _print_first_action(raw_actions)
        drive_hands_from_action(prediction[hand_step], send_hands, left_port, right_port, hand_step)
        print(f"Policy timing: {response.get('policy_timing', {})}")
        print(f"Server timing: {response.get('server_timing', {})}")
        print(f"Comparison metrics: {_metrics(prediction, target)}")

        all_predictions.append(prediction)
        all_targets.append(target)

    stacked_predictions = np.stack(all_predictions)
    stacked_targets = np.stack(all_targets)
    print("=" * 100)
    print(f"Aggregate recorded-data metrics: {_metrics(stacked_predictions, stacked_targets)}")
    print("\nFor real-robot use, replace _make_h2_observation_from_dataset with live uint8 HWC images")
    print("and a live float32 29-value joint state in exactly the same camera keys and state ordering.")
    print("Do not directly execute this 50-step output block: apply limits and execute only 1-3 steps per control cycle.")


if __name__ == "__main__":
    tyro.cli(main)
