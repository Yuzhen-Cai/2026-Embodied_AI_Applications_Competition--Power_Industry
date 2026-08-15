"""Test an H2 WebSocket policy server with recorded local training samples.

This script sends observations to a policy server but never sends returned actions
to a robot and does not write output files.
"""

import time

import numpy as np
import tyro

from openpi_client import websocket_client_policy
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
) -> None:
    """Send recorded H2 observations to a policy server and print its responses."""
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
    print("Safety: this client only sends recorded observations and never commands a robot.\n")

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
        print("Returned fields:")
        print("  actions: policy output action chunk; use only dimensions 0-28 for the 29 H2 joints.")
        print("  state: diagnostic state returned by the policy transform; it is not a command.")
        print("  policy_timing.infer_ms: model sampling time measured inside the policy process.")
        print("  server_timing.infer_ms: total server-side policy inference time.")
        print("  server_timing.prev_total_ms: previous request's server receive/infer/send time, when available.")
        print(f"  client_round_trip_ms: {client_round_trip_ms:.1f} ms, including serialization and network transfer.")

        raw_actions = np.asarray(response["actions"], dtype=np.float32)
        expected_shape = (config.model.action_horizon, config.model.action_dim)
        if raw_actions.shape != expected_shape:
            raise ValueError(f"Unexpected server action shape {raw_actions.shape}; expected {expected_shape}.")
        prediction = raw_actions[:, :29]
        target = np.asarray(sample["action"], dtype=np.float32)

        _print_first_action(raw_actions)
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