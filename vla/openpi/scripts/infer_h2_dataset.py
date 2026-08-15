"""Inspect H2 policy predictions on recorded local training samples.

This script never communicates with a robot and does not write output files.
"""

import pathlib
import time

import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as training_config
from openpi.training import data_loader


def _make_h2_observation(sample: dict) -> dict:
    """Convert one recorded H2 sample to the policy inference input."""
    return {
        "images": {
            "cam_high": sample["observation.images.head"],
            "cam_left_wrist": sample["observation.images.left"],
            "cam_right_wrist": sample["observation.images.right"],
        },
        "state": sample["observation.state"],
        "prompt": sample["task"],
    }


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Calculate action-chunk errors for arrays shaped (samples, steps, joints)."""
    errors = prediction - target
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "first_action_mae": float(np.mean(np.abs(errors[:, 0, :]))),
        "first_action_rmse": float(np.sqrt(np.mean(np.square(errors[:, 0, :])))),
    }


def _print_array(name: str, values: np.ndarray) -> None:
    """Print an entire numeric array at a readable precision."""
    print(f"{name} shape={values.shape}")
    print(np.array2string(values, precision=5, suppress_small=True, threshold=np.inf, max_line_width=160))


def _print_image_summary(images: dict[str, np.ndarray]) -> None:
    """Print metadata for image inputs without dumping pixel values."""
    for camera_name, image in images.items():
        image_array = np.asarray(image)
        print(
            f"  {camera_name}: shape={image_array.shape}, dtype={image_array.dtype}, "
            f"range=[{image_array.min():.0f}, {image_array.max():.0f}]"
        )


def main(
    config_name: str = "pi05_h2_one_hand",
    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/pi05_h2_one_hand/smoke_test/19999"),
    sample_indices: tuple[int, ...] = (0, 1000, 5000, 10000, 15000),
) -> None:
    """Run terminal-only inference against recorded H2 training samples."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not sample_indices:
        raise ValueError("At least one sample index is required.")

    config = training_config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    invalid_indices = [index for index in sample_indices if index < 0 or index >= len(dataset)]
    if invalid_indices:
        raise IndexError(f"Sample indices outside dataset range [0, {len(dataset) - 1}]: {invalid_indices}")

    print(f"Config: {config_name}")
    print(f"Checkpoint: {checkpoint_dir.resolve()}")
    print(f"Dataset: {data_config.data_root} ({len(dataset)} samples)")
    print(f"Requested samples: {list(sample_indices)}")
    print("This script only reads recorded data and never sends commands to a robot.\n")

    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    predictions, targets = [], []

    for index in sample_indices:
        sample = dataset[index]
        observation = _make_h2_observation(sample)
        print("=" * 100)
        print(f"Dataset sample {index}")
        print(f"Task: {observation['prompt']}")
        print("Image inputs:")
        _print_image_summary(observation["images"])
        _print_array("Input state (29 H2 joint positions, rad)", np.asarray(observation["state"]))

        start_time = time.perf_counter()
        result = policy.infer(observation)
        wall_time_ms = (time.perf_counter() - start_time) * 1000
        raw_prediction = np.asarray(result["actions"], dtype=np.float32)
        target = np.asarray(sample["action"], dtype=np.float32)

        expected_raw_shape = (config.model.action_horizon, config.model.action_dim)
        expected_target_shape = (config.model.action_horizon, 29)
        if raw_prediction.shape != expected_raw_shape:
            raise ValueError(f"Unexpected prediction shape {raw_prediction.shape}; expected {expected_raw_shape}.")
        if target.shape != expected_target_shape:
            raise ValueError(f"Unexpected target shape {target.shape}; expected {expected_target_shape}.")

        prediction = raw_prediction[:, :29]
        absolute_error = np.abs(prediction - target)
        sample_metrics = _metrics(prediction[None, ...], target[None, ...])
        print(f"Policy inference time: {result['policy_timing']['infer_ms']:.1f} ms")
        print(f"Wall-clock inference time: {wall_time_ms:.1f} ms")
        _print_array("Raw model actions (50 steps, 32 dimensions)", raw_prediction)
        _print_array("Predicted H2 actions (first 29 dimensions, rad)", prediction)
        _print_array("Recorded target H2 actions (29 dimensions, rad)", target)
        _print_array("Absolute error (rad)", absolute_error)
        _print_array("Raw dimensions 29-31 (padding, not H2 joints)", raw_prediction[:, 29:])
        print("Sample metrics:")
        for name, value in sample_metrics.items():
            print(f"  {name}: {value:.6f}")

        predictions.append(prediction)
        targets.append(target)

    aggregate_metrics = _metrics(np.stack(predictions), np.stack(targets))
    print("=" * 100)
    print("Aggregate metrics")
    for name, value in aggregate_metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    tyro.cli(main)