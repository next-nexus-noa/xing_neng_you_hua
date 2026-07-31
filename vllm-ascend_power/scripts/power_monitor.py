#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sample NPU power and summarize vLLM benchmark energy metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True


def _number(text: Any) -> float | None:
    if text is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(text))
    return float(match.group(0)) if match else None


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_npu_smi_power(output: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    header: list[str] | None = None

    for line in output.splitlines():
        if "|" not in line:
            continue
        cells = _split_table_row(line)
        lower = [cell.lower() for cell in cells]
        if any("power" in cell for cell in lower) and any(
            "npu" in cell or "id" == cell for cell in lower
        ):
            header = lower
            continue
        if not header or len(cells) != len(header):
            continue

        power_idx = next((i for i, cell in enumerate(header) if "power" in cell), -1)
        npu_idx = next(
            (
                i
                for i, cell in enumerate(header)
                if "npu" in cell or cell in ("id", "device")
            ),
            0,
        )
        if power_idx < 0:
            continue
        power_w = _number(cells[power_idx])
        if power_w is None:
            continue
        npu_match = re.search(r"\d+", cells[npu_idx])
        samples.append(
            {
                "npu_id": int(npu_match.group(0)) if npu_match else cells[npu_idx],
                "power_w": power_w,
            }
        )

    if samples:
        return samples

    current_npu: int | str | None = None
    for line in output.splitlines():
        lower = line.lower()
        if "npu" in lower and "id" in lower:
            match = re.search(r"\d+", line)
            if match:
                current_npu = int(match.group(0))
        if "power" in lower:
            power_w = _number(line)
            if power_w is not None:
                samples.append(
                    {
                        "npu_id": current_npu if current_npu is not None else "unknown",
                        "power_w": power_w,
                    }
                )
    return samples


def sample(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    interval_s = max(args.interval_ms, 1) / 1000.0
    npu_filter = None
    if args.npu_ids:
        npu_filter = {item.strip() for item in args.npu_ids.split(",") if item.strip()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = args.command.split()

    with open(output_path, "a", encoding="utf-8") as f:
        while not _STOP:
            ts = time.time()
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_s,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                print(f"power sampling failed: {exc}", file=sys.stderr)
                time.sleep(interval_s)
                continue

            for item in parse_npu_smi_power(result.stdout):
                if npu_filter is not None and str(item["npu_id"]) not in npu_filter:
                    continue
                record = {
                    "ts": ts,
                    "npu_id": item["npu_id"],
                    "power_w": item["power_w"],
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            f.flush()
            time.sleep(interval_s)
    return 0


def _load_power_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _number(item.get("ts"))
            power_w = _number(item.get("power_w"))
            if ts is not None and power_w is not None:
                samples.append({"ts": ts, "power_w": power_w})
    return samples


def _summarize(samples: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "avg_power_w": None,
        "max_power_w": None,
        "energy_j": None,
        "energy_per_request_j": None,
        "energy_per_output_token_j": None,
        "tokens_per_joule": None,
        "output_tokens_per_second_per_watt": None,
    }
    if not samples:
        return metrics

    by_ts: dict[float, float] = defaultdict(float)
    for item in samples:
        by_ts[float(item["ts"])] += float(item["power_w"])
    timeline = sorted(by_ts.items())
    metrics["max_power_w"] = max(power for _, power in timeline)

    if len(timeline) == 1:
        metrics["avg_power_w"] = timeline[0][1]
        metrics["energy_j"] = 0.0
    else:
        energy_j = 0.0
        for (ts, power), (next_ts, _) in zip(timeline, timeline[1:]):
            energy_j += power * max(0.0, next_ts - ts)
        duration_s = max(0.0, timeline[-1][0] - timeline[0][0])
        metrics["energy_j"] = energy_j
        metrics["avg_power_w"] = energy_j / duration_s if duration_s > 0 else None

    energy_j = metrics["energy_j"]
    avg_power_w = metrics["avg_power_w"]
    completed = result.get("completed")
    total_output_tokens = result.get("total_output_tokens")
    output_throughput = result.get("output_throughput")

    if energy_j and energy_j > 0:
        if completed:
            metrics["energy_per_request_j"] = energy_j / completed
        if total_output_tokens:
            metrics["energy_per_output_token_j"] = energy_j / total_output_tokens
            metrics["tokens_per_joule"] = total_output_tokens / energy_j
    if avg_power_w and output_throughput is not None:
        metrics["output_tokens_per_second_per_watt"] = output_throughput / avg_power_w
    return metrics


def report(args: argparse.Namespace) -> int:
    result: dict[str, Any] = {}
    benchmark_json = Path(args.benchmark_json) if args.benchmark_json else None
    if benchmark_json and benchmark_json.exists():
        with open(benchmark_json, encoding="utf-8-sig") as f:
            result = json.load(f)

    metrics = _summarize(_load_power_samples(Path(args.input)), result)
    result.update(metrics)
    result["power_metrics_file"] = args.input

    for key, value in metrics.items():
        print(f"{key}: {'N/A' if value is None else value}")

    output_path = Path(args.output) if args.output else benchmark_json
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2)
            f.write(os.linesep)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--output", required=True)
    sample_parser.add_argument("--interval-ms", type=int, default=500)
    sample_parser.add_argument("--npu-ids", default=None)
    sample_parser.add_argument("--command", default="npu-smi info")
    sample_parser.add_argument("--timeout-s", type=float, default=5.0)
    sample_parser.set_defaults(func=sample)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--input", required=True)
    report_parser.add_argument("--benchmark-json", default=None)
    report_parser.add_argument("--output", default=None)
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
