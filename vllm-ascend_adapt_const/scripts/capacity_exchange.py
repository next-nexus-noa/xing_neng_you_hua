#!/usr/bin/env python3
"""Publish and consume per-workload QPS records through a shared directory."""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def record_path(root: str, session: str, model_label: str, dataset: str) -> Path:
    session_dir = Path(root) / safe_component(session)
    filename = f"{safe_component(model_label)}__{safe_component(dataset)}.json"
    return session_dir / filename


def publish(args: argparse.Namespace) -> None:
    with open(args.benchmark_json, encoding="utf-8-sig") as f:
        benchmark = json.load(f)

    capacity = float(benchmark.get("request_throughput", 0))
    completed = int(benchmark.get("completed", 0))
    failed = int(benchmark.get("failed", 0))
    if capacity <= 0 or completed <= 0 or failed != 0:
        raise SystemExit(
            f"invalid capacity probe: throughput={capacity}, "
            f"completed={completed}, failed={failed}"
        )

    record = {
        "status": "ready",
        "session": args.session,
        "model_label": args.model_label,
        "dataset": args.dataset,
        "capacity_qps": capacity,
        "low_qps": capacity * args.low_factor,
        "medium_qps": capacity * args.medium_factor,
        "high_qps": capacity * args.high_factor,
        "completed": completed,
        "failed": failed,
        "probe_rate": args.probe_rate,
        "num_prompts": args.num_prompts,
        "run_id": args.run_id,
        "published_at_unix_s": time.time(),
    }

    destination = record_path(
        args.exchange_dir, args.session, args.model_label, args.dataset
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise

    print(
        f"Published {args.model_label}/{args.dataset}: "
        f"C={capacity:.6g} -> {destination}"
    )


def wait_for_rate(args: argparse.Namespace) -> None:
    source = record_path(
        args.exchange_dir, args.session, args.model_label, args.dataset
    )
    rate_key = {
        "low": "low_qps",
        "medium": "medium_qps",
        "high": "high_qps",
    }[args.level]
    deadline = time.monotonic() + args.timeout
    next_progress = time.monotonic()
    last_error = ""

    while True:
        now = time.monotonic()
        if now >= next_progress:
            print(f"Waiting for capacity record: {source}", file=sys.stderr)
            next_progress = now + 60
        if source.is_file():
            try:
                with source.open(encoding="utf-8") as f:
                    record = json.load(f)
                if record.get("status") != "ready":
                    raise ValueError(f"status={record.get('status')!r}")
                if record.get("session") != args.session:
                    raise ValueError("session mismatch")
                if record.get("model_label") != args.model_label:
                    raise ValueError("model_label mismatch")
                if record.get("dataset") != args.dataset:
                    raise ValueError("dataset mismatch")
                if int(record.get("failed", -1)) != 0:
                    raise ValueError(f"failed={record.get('failed')}")
                value = float(record[rate_key])
                if value <= 0:
                    raise ValueError(f"{rate_key}={value}")
                print(f"{value:.9g}")
                return
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last_error = str(exc)

        if now >= deadline:
            detail = f"; last error: {last_error}" if last_error else ""
            raise SystemExit(f"timed out waiting for {source}{detail}")
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--benchmark-json", required=True)
    publish_parser.add_argument("--exchange-dir", required=True)
    publish_parser.add_argument("--session", required=True)
    publish_parser.add_argument("--model-label", required=True)
    publish_parser.add_argument("--dataset", required=True)
    publish_parser.add_argument("--probe-rate", required=True)
    publish_parser.add_argument("--num-prompts", required=True, type=int)
    publish_parser.add_argument("--run-id", required=True)
    publish_parser.add_argument("--low-factor", type=float, default=0.35)
    publish_parser.add_argument("--medium-factor", type=float, default=1.0)
    publish_parser.add_argument("--high-factor", type=float, default=2.0)
    publish_parser.set_defaults(func=publish)

    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("--exchange-dir", required=True)
    wait_parser.add_argument("--session", required=True)
    wait_parser.add_argument("--model-label", required=True)
    wait_parser.add_argument("--dataset", required=True)
    wait_parser.add_argument("--level", choices=("low", "medium", "high"), required=True)
    wait_parser.add_argument("--timeout", type=float, default=7200)
    wait_parser.add_argument("--interval", type=float, default=2)
    wait_parser.set_defaults(func=wait_for_rate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
