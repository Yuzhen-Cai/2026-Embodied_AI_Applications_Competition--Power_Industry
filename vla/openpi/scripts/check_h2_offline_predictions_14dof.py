"""Evaluate an H2 checkpoint on recorded local LeRobot samples without robot control."""

import csv
import json
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as training_config
from openpi.training import data_loader

ACTION_DIM = 14
JOINT_GROUPS = {
    "Left arm": slice(0, 7),
    "Right arm": slice(7, 14),
}


def _save_error_summary_plots(mean_error_by_step_joint: np.ndarray, output_dir: pathlib.Path) -> list[pathlib.Path]:
    """Save aggregate error plots from a mean time-step-by-joint error matrix."""
    mean_error_by_joint = mean_error_by_step_joint.mean(axis=0)
    paths = []

    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    image = axis.imshow(mean_error_by_step_joint.T, aspect="auto", cmap="magma")
    axis.set_xlabel("Action step")
    axis.set_ylabel("Joint dimension")
    axis.set_title("Mean absolute error by action step and joint")
    figure.colorbar(image, ax=axis, label="Absolute error (rad)")
    path = output_dir / "mean_absolute_error_heatmap.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    joints = np.arange(mean_error_by_joint.size)
    axis.bar(joints, mean_error_by_joint, color="#276FBF")
    for group_name, joint_slice in JOINT_GROUPS.items():
        midpoint = (joint_slice.start + joint_slice.stop - 1) / 2
        axis.text(midpoint, 1.02, group_name, ha="center", va="bottom", transform=axis.get_xaxis_transform())
        if joint_slice.stop < mean_error_by_joint.size:
            axis.axvline(joint_slice.stop - 0.5, color="#9AA5B1", linewidth=0.8)
    axis.set_xlabel("Joint dimension")
    axis.set_ylabel("Mean absolute error (rad)")
    axis.set_title("Mean absolute error by joint")
    axis.set_xticks(joints)
    path = output_dir / "mean_absolute_error_by_joint.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def _save_episode_error_plot(episode_rows: list[dict], output_dir: pathlib.Path) -> pathlib.Path:
    """Visualize whole-episode chunk and first-action errors, highlighting the worst episodes."""
    episode_ids = np.asarray([row["episode_index"] for row in episode_rows])
    chunk_mae = np.asarray([row["mae"] for row in episode_rows])
    first_action_mae = np.asarray([row["first_action_mae"] for row in episode_rows])
    worst_indices = np.argsort(first_action_mae)[-min(10, len(episode_rows)) :]

    figure, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
    axes[0].plot(episode_ids, chunk_mae, marker="o", markersize=3, color="#276FBF", label="Chunk MAE")
    axes[0].scatter(episode_ids[worst_indices], chunk_mae[worst_indices], color="#D1495B", zorder=3, label="Top 10 first-action errors")
    axes[0].set_ylabel("Chunk MAE (rad)")
    axes[0].set_title("Full-training-data error by episode")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(episode_ids, first_action_mae, marker="o", markersize=3, color="#2A9D8F", label="First-action MAE")
    axes[1].scatter(
        episode_ids[worst_indices], first_action_mae[worst_indices], color="#D1495B", zorder=3, label="Top 10 first-action errors"
    )
    for index in worst_indices:
        axes[1].annotate(str(episode_ids[index]), (episode_ids[index], first_action_mae[index]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_xlabel("Episode index")
    axes[1].set_ylabel("First-action MAE (rad)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    path = output_dir / "error_by_episode.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _save_sample_plot(prediction: np.ndarray, target: np.ndarray, index: int, output_dir: pathlib.Path) -> pathlib.Path:
    """Save target-versus-prediction trajectories and absolute errors for one sample."""
    steps = np.arange(prediction.shape[0])
    figure, axes = plt.subplots(2, len(JOINT_GROUPS), figsize=(22, 8), sharex="col", constrained_layout=True)
    for column, (group_name, joint_slice) in enumerate(JOINT_GROUPS.items()):
        trajectory_axis = axes[0, column]
        error_axis = axes[1, column]
        for joint in range(joint_slice.start, joint_slice.stop):
            label = f"j{joint}"
            trajectory_axis.plot(steps, target[:, joint], linewidth=1.5, label=f"target {label}")
            trajectory_axis.plot(steps, prediction[:, joint], "--", linewidth=1.1, label=f"prediction {label}")
            error_axis.plot(steps, np.abs(prediction[:, joint] - target[:, joint]), linewidth=1.3, label=label)
        trajectory_axis.set_title(group_name)
        trajectory_axis.set_ylabel("Joint position (rad)")
        trajectory_axis.grid(alpha=0.25)
        trajectory_axis.legend(fontsize=6, ncol=2)
        error_axis.set_xlabel("Action step")
        error_axis.set_ylabel("Absolute error (rad)")
        error_axis.grid(alpha=0.25)
        error_axis.legend(fontsize=7, ncol=2)
    figure.suptitle(f"H2 offline action prediction, dataset sample {index}", fontsize=15)
    path = output_dir / f"sample_{index:05d}_action_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _make_h2_observation(sample: dict) -> dict:
    """Convert a raw LeRobot H2 sample to the H2 policy's inference input."""
    return {
        "images": {
            "cam_high": sample["observation.images.head"],
            "cam_left_wrist": sample["observation.images.left"],
            "cam_right_wrist": sample["observation.images.right"],
        },
        "state": sample["observation.state"],
        "prompt": sample["task"],
    }


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "first_action_mae": float(np.mean(np.abs(errors[:, 0, :]))),
        "first_action_rmse": float(np.sqrt(np.mean(np.square(errors[:, 0, :])))),
    }


def _episode_metrics(episode_stats: dict[int, dict]) -> list[dict]:
    """Convert streaming per-episode sums into comparable metrics."""
    rows = []
    for episode_index, stats in sorted(episode_stats.items()):
        sample_count = stats["sample_count"]
        chunk_values = sample_count * 50 * ACTION_DIM
        first_action_values = sample_count * ACTION_DIM
        rows.append(
            {
                "episode_index": episode_index,
                "sample_count": sample_count,
                "prompt": stats["prompt"],
                "mae": stats["absolute_error_sum"] / chunk_values,
                "rmse": float(np.sqrt(stats["squared_error_sum"] / chunk_values)),
                "max_abs_error": stats["max_abs_error"],
                "first_action_mae": stats["first_absolute_error_sum"] / first_action_values,
                "first_action_rmse": float(np.sqrt(stats["first_squared_error_sum"] / first_action_values)),
                "mean_infer_ms": stats["infer_ms_sum"] / sample_count,
            }
        )
    return rows


def main(
    config_name: str = "pi05_h2_one_hand",
    checkpoint_dir: pathlib.Path | None = None,
    sample_indices: tuple[int, ...] = (),
    num_episodes: int | None = None,
    all_samples: bool = False,  # noqa: FBT001, FBT002
    top_k: int = 20,
    output_dir: pathlib.Path | None = None,
) -> None:
    """Run checkpoint inference against recorded H2 action chunks and save error reports.

    Specify one of sample_indices, num_episodes, or all_samples. With num_episodes, one
    center frame is selected from each uniformly spaced LeRobot v3 episode. all_samples
    evaluates every frame, aggregates metrics by episode, and saves the worst samples.
    Without a selection option, the historical five fixed sample indices are used.
    """
    if checkpoint_dir is None:
        checkpoint_dir = pathlib.Path("checkpoints") / config_name / "smoke_test" / "19999"
    if output_dir is None:
        output_dir = pathlib.Path("visualization_output") / f"{config_name}_offline_predictions"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    config = training_config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")
    selection_count = int(bool(sample_indices)) + int(num_episodes is not None) + int(all_samples)
    if selection_count > 1:
        raise ValueError("Specify only one of sample_indices, num_episodes, or all_samples.")
    if all_samples:
        if not isinstance(dataset, data_loader.LeRobotV3Dataset):
            raise TypeError("Full-dataset evaluation is supported only for local LeRobot v3 datasets.")
        episode_samples = dataset.all_episode_samples()
    elif num_episodes is not None:
        if not isinstance(dataset, data_loader.LeRobotV3Dataset):
            raise TypeError("Episode-uniform sampling is supported only for local LeRobot v3 datasets.")
        episode_samples = dataset.uniform_episode_samples(num_episodes)
    else:
        selected_indices = sample_indices or (0, 1000, 5000, 10000, 15000)
        episode_samples = [(None, index) for index in selected_indices]

    invalid_indices = [index for _, index in episode_samples if index < 0 or index >= len(dataset)]
    if invalid_indices:
        raise IndexError(f"Sample indices outside dataset range [0, {len(dataset) - 1}]: {invalid_indices}")

    output_dir.mkdir(parents=True, exist_ok=True)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    predictions, targets, rows, sample_plots = [], [], [], []
    absolute_error_by_step_joint_sum = np.zeros((config.model.action_horizon, ACTION_DIM), dtype=np.float64)
    aggregate_absolute_error_sum = 0.0
    aggregate_squared_error_sum = 0.0
    aggregate_first_absolute_error_sum = 0.0
    aggregate_first_squared_error_sum = 0.0
    aggregate_max_abs_error = 0.0
    infer_ms_sum = 0.0
    episode_stats: dict[int, dict] = {}

    for sample_number, (episode_index, index) in enumerate(episode_samples, start=1):
        sample = dataset[index]
        start_time = time.perf_counter()
        result = policy.infer(_make_h2_observation(sample))
        wall_time_ms = (time.perf_counter() - start_time) * 1000

        prediction = np.asarray(result["actions"], dtype=np.float32)
        target = np.asarray(sample["action"], dtype=np.float32)
        if prediction.shape != (config.model.action_horizon, config.model.action_dim):
            raise ValueError(f"Unexpected prediction shape {prediction.shape} for sample {index}.")
        if target.shape != (config.model.action_horizon, ACTION_DIM):
            raise ValueError(f"Unexpected target shape {target.shape} for sample {index}.")

        prediction = prediction[:, :ACTION_DIM]
        sample_metrics = _metrics(prediction[None, ...], target[None, ...])
        absolute_errors = np.abs(prediction - target)
        squared_errors = np.square(prediction - target)
        absolute_error_by_step_joint_sum += absolute_errors
        aggregate_absolute_error_sum += float(absolute_errors.sum())
        aggregate_squared_error_sum += float(squared_errors.sum())
        aggregate_first_absolute_error_sum += float(absolute_errors[0].sum())
        aggregate_first_squared_error_sum += float(squared_errors[0].sum())
        aggregate_max_abs_error = max(aggregate_max_abs_error, float(absolute_errors.max()))
        infer_ms = float(result["policy_timing"]["infer_ms"])
        infer_ms_sum += infer_ms
        rows.append(
            {
                "sample_index": index,
                "episode_index": episode_index,
                "prompt": sample["task"],
                "infer_ms": infer_ms,
                "wall_time_ms": wall_time_ms,
                **sample_metrics,
            }
        )
        if all_samples:
            stats = episode_stats.setdefault(
                episode_index,
                {
                    "sample_count": 0,
                    "prompt": sample["task"],
                    "absolute_error_sum": 0.0,
                    "squared_error_sum": 0.0,
                    "max_abs_error": 0.0,
                    "first_absolute_error_sum": 0.0,
                    "first_squared_error_sum": 0.0,
                    "infer_ms_sum": 0.0,
                },
            )
            stats["sample_count"] += 1
            stats["absolute_error_sum"] += float(absolute_errors.sum())
            stats["squared_error_sum"] += float(squared_errors.sum())
            stats["max_abs_error"] = max(stats["max_abs_error"], float(absolute_errors.max()))
            stats["first_absolute_error_sum"] += float(absolute_errors[0].sum())
            stats["first_squared_error_sum"] += float(squared_errors[0].sum())
            stats["infer_ms_sum"] += infer_ms
        else:
            predictions.append(prediction)
            targets.append(target)
            sample_plots.append(_save_sample_plot(prediction, target, index, output_dir))
        episode_text = f"episode={episode_index} " if episode_index is not None else ""
        if not all_samples or sample_number % 100 == 0 or sample_number == len(episode_samples):
            print(
                f"{episode_text}sample={index} ({sample_number}/{len(episode_samples)}) infer_ms={infer_ms:.1f} "
                f"first_action_mae={sample_metrics['first_action_mae']:.4f} rad "
                f"chunk_mae={sample_metrics['mae']:.4f} rad"
            )

    total_samples = len(rows)
    total_chunk_values = total_samples * config.model.action_horizon * ACTION_DIM
    total_first_action_values = total_samples * ACTION_DIM
    aggregate_metrics = {
        "mae": aggregate_absolute_error_sum / total_chunk_values,
        "rmse": float(np.sqrt(aggregate_squared_error_sum / total_chunk_values)),
        "max_abs_error": aggregate_max_abs_error,
        "first_action_mae": aggregate_first_absolute_error_sum / total_first_action_values,
        "first_action_rmse": float(np.sqrt(aggregate_first_squared_error_sum / total_first_action_values)),
    }
    mean_error_by_step_joint = absolute_error_by_step_joint_sum / total_samples
    summary_plots = _save_error_summary_plots(mean_error_by_step_joint, output_dir)
    episode_rows = _episode_metrics(episode_stats) if all_samples else []
    high_error_rows = sorted(rows, key=lambda row: (row["first_action_mae"], row["mae"]), reverse=True)[:top_k]
    if episode_rows:
        summary_plots.append(_save_episode_error_plot(episode_rows, output_dir))
    summary = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "config_name": config_name,
        "dataset_root": data_config.data_root,
        "sample_indices": [index for _, index in episode_samples],
        "episode_indices": [episode_index for episode_index, _ in episode_samples],
        "action_horizon": config.model.action_horizon,
        "action_dim": ACTION_DIM,
        "all_samples": all_samples,
        "evaluated_sample_count": total_samples,
        "metrics": aggregate_metrics,
        "mean_infer_ms": infer_ms_sum / total_samples,
        "high_error_selection": {"sort_key": "first_action_mae, then mae", "top_k": top_k},
        "visualizations": [str(path) for path in [*summary_plots, *sample_plots]],
    }

    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    with (output_dir / "per_sample_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "high_error_samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=high_error_rows[0].keys())
        writer.writeheader()
        writer.writerows(high_error_rows)
    if episode_rows:
        with (output_dir / "per_episode_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=episode_rows[0].keys())
            writer.writeheader()
            writer.writerows(episode_rows)
    else:
        predictions_array = np.stack(predictions)
        targets_array = np.stack(targets)
        np.savez_compressed(output_dir / "action_chunks.npz", predictions=predictions_array, targets=targets_array)

    print(f"\nSaved summary: {output_dir / 'summary.json'}")
    print(f"Saved per-sample metrics: {output_dir / 'per_sample_metrics.csv'}")
    print(f"Saved high-error samples: {output_dir / 'high_error_samples.csv'}")
    if episode_rows:
        print(f"Saved per-episode metrics: {output_dir / 'per_episode_metrics.csv'}")
        print("Skipped action_chunks.npz for full-dataset evaluation to avoid a very large output file.")
    else:
        print(f"Saved action chunks: {output_dir / 'action_chunks.npz'}")
    print(f"Saved visualizations: {output_dir}")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    tyro.cli(main)
