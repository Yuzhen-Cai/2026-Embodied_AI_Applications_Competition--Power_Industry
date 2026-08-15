"""Measure H2 WebSocket policy inference latency with one repeated recorded observation.

This script never sends actions to the robot or serial ports. It loads one local
recorded sample once, sends the same observation repeatedly, and reports policy,
server, and client round-trip latency statistics.
"""

import time

import numpy as np
from openpi_client import websocket_client_policy
import tyro

from openpi.training import config as training_config
from openpi.training import data_loader


def _make_h2_observation(sample: dict) -> dict:
    """Map a recorded H2 sample to the policy server observation contract."""
    return {
        "images": {
            "cam_high": sample["observation.images.head"],
            "cam_left_wrist": sample["observation.images.left"],
            "cam_right_wrist": sample["observation.images.right"],
        },
        "state": sample["observation.state"],
        "prompt": sample["task"],
    }


def _latency_stats(samples_ms: list[float]) -> dict[str, float]:
    """Return latency statistics in milliseconds."""
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def _print_latency_stats(title: str, samples_ms: list[float]) -> dict[str, float]:
    stats = _latency_stats(samples_ms)
    print(
        f"{title}: mean={stats['mean_ms']:.2f} ms, std={stats['std_ms']:.2f} ms, "
        f"p50={stats['p50_ms']:.2f} ms, p95={stats['p95_ms']:.2f} ms, "
        f"p99={stats['p99_ms']:.2f} ms, min={stats['min_ms']:.2f} ms, max={stats['max_ms']:.2f} ms"
    )
    return stats


def main(
    host: str = "127.0.0.1",
    port: int = 9000,
    config_name: str = "pi05_h2_one_hand_mixu",
    sample_index: int = 145,
    requests: int = 100,
    warmup_requests: int = 5,
) -> None:
    """Send one recorded observation repeatedly and report WebSocket latency.

    The warm-up requests are excluded from the reported metrics. The estimated
    transport/serialization time is client round-trip time minus server inference
    time, so it is not a pure network-only measurement.
    """
    if requests < 1:
        raise ValueError(f"requests must be at least 1, got {requests}.")
    if warmup_requests < 0:
        raise ValueError(f"warmup_requests must be non-negative, got {warmup_requests}.")

    config = training_config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index must be in [0, {len(dataset) - 1}], got {sample_index}.")

    sample = dataset[sample_index]
    observation = _make_h2_observation(sample)
    print(f"Connecting to H2 policy server: ws://{host}:{port}")
    print(f"Using one cached recorded observation: sample={sample_index}, prompt={observation['prompt']!r}")
    print(f"Warm-up requests: {warmup_requests}; measured requests: {requests}")
    print("Safety: this benchmark never sends returned actions to a robot or serial port.")

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"Server metadata: {client.get_server_metadata()}")

    for request_number in range(1, warmup_requests + 1):
        client.infer(observation)
        print(f"Completed warm-up request {request_number}/{warmup_requests}")

    round_trip_ms: list[float] = []
    policy_infer_ms: list[float] = []
    server_infer_ms: list[float] = []
    estimated_transport_and_serialization_ms: list[float] = []

    for request_number in range(1, requests + 1):
        start_time = time.perf_counter()
        response = client.infer(observation)
        round_trip = (time.perf_counter() - start_time) * 1000
        policy_infer = float(response["policy_timing"]["infer_ms"])
        server_infer = float(response["server_timing"]["infer_ms"])

        round_trip_ms.append(round_trip)
        policy_infer_ms.append(policy_infer)
        server_infer_ms.append(server_infer)
        estimated_transport_and_serialization_ms.append(max(0.0, round_trip - server_infer))

        if request_number % 10 == 0 or request_number == requests:
            print(
                f"request={request_number}/{requests} policy={policy_infer:.2f} ms "
                f"server={server_infer:.2f} ms round_trip={round_trip:.2f} ms"
            )

    print("\nMeasured latency summary (warm-up excluded):")
    policy_stats = _print_latency_stats("Policy inference", policy_infer_ms)
    server_stats = _print_latency_stats("Server inference", server_infer_ms)
    round_trip_stats = _print_latency_stats("Client round trip", round_trip_ms)
    _print_latency_stats("Estimated transport + serialization", estimated_transport_and_serialization_ms)
    print(f"Client-observed throughput: {1000 / round_trip_stats['mean_ms']:.2f} requests/s")
    print(
        "Note: estimated transport + serialization includes WebSocket serialization, scheduling, and network transfer; "
        "it is not a pure network ping measurement."
    )
    print(
        f"Server-side overhead beyond model sampling: {server_stats['mean_ms'] - policy_stats['mean_ms']:.2f} ms on average."
    )


if __name__ == "__main__":
    tyro.cli(main)
