# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Power metric helpers for benchmark result post-processing."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


POWER_METRIC_KEYS = (
    "avg_power_w",
    "max_power_w",
    "energy_j",
    "energy_per_request_j",
    "energy_per_output_token_j",
    "tokens_per_joule",
    "output_tokens_per_second_per_watt",
)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None


def load_power_samples(path: str | Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _to_float(item.get("ts"))
            power_w = _to_float(item.get("power_w"))
            if ts is None or power_w is None:
                continue
            samples.append(
                {
                    "ts": ts,
                    "npu_id": item.get("npu_id", "unknown"),
                    "power_w": power_w,
                }
            )
    return samples


def summarize_power_samples(
    samples: list[dict[str, Any]],
    completed: int | None = None,
    total_output_tokens: int | None = None,
    output_throughput: float | None = None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = dict.fromkeys(POWER_METRIC_KEYS)
    if not samples:
        return metrics

    by_ts: dict[float, float] = defaultdict(float)
    for sample in samples:
        by_ts[float(sample["ts"])] += float(sample["power_w"])

    timeline = sorted(by_ts.items())
    max_power_w = max(power for _, power in timeline)

    if len(timeline) == 1:
        avg_power_w = timeline[0][1]
        energy_j = 0.0
    else:
        energy_j = 0.0
        for (ts, power), (next_ts, _) in zip(timeline, timeline[1:]):
            interval_s = max(0.0, next_ts - ts)
            energy_j += power * interval_s
        duration_s = max(0.0, timeline[-1][0] - timeline[0][0])
        avg_power_w = energy_j / duration_s if duration_s > 0 else None

    metrics["avg_power_w"] = avg_power_w
    metrics["max_power_w"] = max_power_w
    metrics["energy_j"] = energy_j

    if energy_j > 0:
        if completed:
            metrics["energy_per_request_j"] = energy_j / completed
        if total_output_tokens:
            metrics["energy_per_output_token_j"] = energy_j / total_output_tokens
            metrics["tokens_per_joule"] = total_output_tokens / energy_j

    if avg_power_w and output_throughput is not None:
        metrics["output_tokens_per_second_per_watt"] = (
            output_throughput / avg_power_w
        )

    return metrics


def calculate_power_metrics(
    power_metrics_file: str | Path,
    benchmark_result: dict[str, Any],
) -> dict[str, float | None]:
    samples = load_power_samples(power_metrics_file)
    return summarize_power_samples(
        samples,
        completed=benchmark_result.get("completed"),
        total_output_tokens=benchmark_result.get("total_output_tokens"),
        output_throughput=benchmark_result.get("output_throughput"),
    )


def print_power_metrics(metrics: dict[str, float | None]) -> None:
    print("{s:{c}^{n}}".format(s=" Power Metrics ", n=50, c="-"))
    labels = {
        "avg_power_w": "Average power (W):",
        "max_power_w": "Max power (W):",
        "energy_j": "Energy (J):",
        "energy_per_request_j": "Energy per request (J/req):",
        "energy_per_output_token_j": "Energy per output token (J/tok):",
        "tokens_per_joule": "Output tokens per joule (tok/J):",
        "output_tokens_per_second_per_watt": "Output tok/s/W:",
    }
    for key in POWER_METRIC_KEYS:
        value = metrics.get(key)
        text = "N/A" if value is None else f"{value:.6g}"
        print("{:<40} {:<10}".format(labels[key], text))
