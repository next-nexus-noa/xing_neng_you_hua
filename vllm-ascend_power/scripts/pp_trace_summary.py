#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stage_total(record: dict[str, Any], prefix: str) -> float:
    return sum(
        e["duration_ms"]
        for e in record.get("events", [])
        if e["stage"] == prefix or e["stage"].startswith(prefix + ".")
    )


def stage_events(record: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    return [
        e for e in record.get("events", [])
        if e["stage"] == stage or e["stage"].startswith(stage + ".")
    ]


def union_intervals(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals = sorted(
        (e["start_ms"], e["end_ms"])
        for e in events
        if e["end_ms"] > e["start_ms"]
    )
    if not intervals:
        return []
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def intervals_total_ms(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def union_ms(events: list[dict[str, Any]]) -> float:
    return intervals_total_ms(union_intervals(events))


def interval_intersection_ms(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    total = 0.0
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def build_dual_metrics_from_timelines(
    timelines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in timelines:
        if rec.get("pp_size") == 2:
            by_step[int(rec["step"])].append(rec)

    metrics: list[dict[str, Any]] = []
    for step, records in sorted(by_step.items()):
        if len(records) != 2:
            continue
        records = sorted(records, key=lambda r: r["pp_rank"])
        all_events = [e for r in records for e in r.get("events", [])]
        if not all_events:
            continue
        step_start = min(e["start_ms"] for e in all_events)
        step_end = max(e["end_ms"] for e in all_events)
        makespan = max(0.0, step_end - step_start)
        if makespan == 0.0:
            continue
        compute_intervals = [
            union_intervals(stage_events(r, "runner.forward")) for r in records
        ]
        compute_ms = [intervals_total_ms(intervals) for intervals in compute_intervals]
        compute_sum_ms = [stage_total(r, "runner.forward") for r in records]
        active_ms = [union_ms(r.get("events", [])) for r in records]
        overlap = interval_intersection_ms(
            compute_intervals[0], compute_intervals[1]
        )
        total_compute = sum(compute_ms)
        item = {
            "type": "dual_pp_metrics",
            "step": step,
            "pp_size": 2,
            "total_tokens": records[0].get("total_tokens"),
            "makespan_ms": makespan,
            "rank_active_ms": active_ms,
            "rank_compute_ms": compute_ms,
            "rank_compute_sum_ms": compute_sum_ms,
            "rank_compute_utilization": [v / makespan for v in compute_ms],
            "rank_active_utilization": [v / makespan for v in active_ms],
            "bubble_ms": max(0.0, 2 * makespan - total_compute),
            "bubble_ratio": max(0.0, 2 * makespan - total_compute)
            / (2 * makespan),
            "compute_overlap_ms": overlap,
            "compute_overlap_ratio": overlap / min(compute_ms)
            if min(compute_ms) > 0
            else 0.0,
        }
        npu_compute_ms = [
            sum(
                float(e["npu_duration_ms"])
                for e in stage_events(r, "runner.forward")
                if "npu_duration_ms" in e
            )
            for r in records
        ]
        if any(v > 0 for v in npu_compute_ms):
            item["rank_npu_compute_ms"] = npu_compute_ms
        metrics.append(item)
    return metrics


def load_trace(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timelines: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at line {line_no}: {exc}") from exc
            if rec.get("type") == "rank_timeline":
                timelines.append(rec)
            elif rec.get("type") == "dual_pp_metrics":
                metrics.append(rec)
    if not metrics:
        metrics = build_dual_metrics_from_timelines(timelines)
    return timelines, metrics


def fmt_ms(v: float) -> str:
    return f"{v:8.3f} ms"


def fmt_pct(v: float) -> str:
    return f"{v * 100:7.2f}%"


def summarize_metrics(metrics: list[dict[str, Any]], skip: int) -> str:
    m = metrics[skip:]
    if not m:
        return "No dual PP metrics after applying --skip.\n"

    makespan = [x["makespan_ms"] for x in m]
    bubble = [x["bubble_ratio"] for x in m]
    overlap = [x["compute_overlap_ratio"] for x in m]
    rank0_compute = [x["rank_compute_utilization"][0] for x in m]
    rank1_compute = [x["rank_compute_utilization"][1] for x in m]
    rank0_active = [x["rank_active_utilization"][0] for x in m]
    rank1_active = [x["rank_active_utilization"][1] for x in m]
    rank0_compute_ms = [x["rank_compute_ms"][0] for x in m]
    rank1_compute_ms = [x["rank_compute_ms"][1] for x in m]
    npu_metrics = [x for x in m if "rank_npu_compute_ms" in x]

    lines = [
        "Dual-card PP summary",
        f"  steps              : {len(m)} "
        f"(step {m[0]['step']}..{m[-1]['step']})",
        f"  tokens range       : {min(x.get('total_tokens') or 0 for x in m)}.."
        f"{max(x.get('total_tokens') or 0 for x in m)}",
        "  makespan avg/p50/p95: "
        f"{fmt_ms(avg(makespan))} / {fmt_ms(percentile(makespan, 0.50))} / "
        f"{fmt_ms(percentile(makespan, 0.95))}",
        "  bubble ratio avg/p50/p95: "
        f"{fmt_pct(avg(bubble))} / {fmt_pct(percentile(bubble, 0.50))} / "
        f"{fmt_pct(percentile(bubble, 0.95))}",
        "  compute overlap avg/max: "
        f"{fmt_pct(avg(overlap))} / {fmt_pct(max(overlap))}",
        "  compute util avg rank0/rank1: "
        f"{fmt_pct(avg(rank0_compute))} / {fmt_pct(avg(rank1_compute))}",
        "  active util avg rank0/rank1 : "
        f"{fmt_pct(avg(rank0_active))} / {fmt_pct(avg(rank1_active))}",
        "  compute ms avg rank0/rank1  : "
        f"{fmt_ms(avg(rank0_compute_ms))} / {fmt_ms(avg(rank1_compute_ms))}",
    ]
    if npu_metrics:
        rank0_npu_compute_ms = [x["rank_npu_compute_ms"][0] for x in npu_metrics]
        rank1_npu_compute_ms = [x["rank_npu_compute_ms"][1] for x in npu_metrics]
        lines.append(
            "  NPU event compute avg rank0/rank1: "
            f"{fmt_ms(avg(rank0_npu_compute_ms))} / "
            f"{fmt_ms(avg(rank1_npu_compute_ms))}"
        )
    if avg(overlap) < 0.05:
        lines.append(
            "  verdict            : low/no compute overlap; this is not "
            "micro-batch pipeline overlap."
        )
    elif avg(overlap) < 0.30:
        lines.append("  verdict            : limited compute overlap.")
    else:
        lines.append("  verdict            : visible compute overlap.")
    return "\n".join(lines) + "\n"


def summarize_stages(timelines: list[dict[str, Any]], skip: int, top: int) -> str:
    by_rank_stage: dict[tuple[int, str], list[float]] = defaultdict(list)
    for rec in timelines:
        if int(rec.get("step", -1)) < skip:
            continue
        pp_rank = int(rec.get("pp_rank", -1))
        for event in rec.get("events", []):
            by_rank_stage[(pp_rank, event["stage"])].append(event["duration_ms"])

    lines = ["Stage duration averages"]
    ranks = sorted({rank for rank, _ in by_rank_stage})
    for rank in ranks:
        rows = []
        for (pp_rank, stage), values in by_rank_stage.items():
            if pp_rank == rank:
                rows.append((avg(values), percentile(values, 0.95), stage, len(values)))
        lines.append(f"  PP rank {rank}:")
        for mean, p95, stage, count in sorted(rows, reverse=True)[:top]:
            lines.append(
                f"    {stage:32s} avg={mean:8.3f} ms "
                f"p95={p95:8.3f} ms n={count}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize vLLM Ascend PP trace JSONL files."
    )
    parser.add_argument("trace_file", type=Path)
    parser.add_argument(
        "--skip",
        type=int,
        default=5,
        help="Skip the first N dual-PP steps as warmup. Default: 5.",
    )
    parser.add_argument(
        "--top-stages",
        type=int,
        default=10,
        help="Print top N stages per PP rank. Default: 10.",
    )
    args = parser.parse_args()

    timelines, metrics = load_trace(args.trace_file)
    print(f"Trace file: {args.trace_file}")
    print(f"Rank timelines: {len(timelines)}")
    print(f"Dual PP metrics: {len(metrics)}")
    print()
    print(summarize_metrics(metrics, args.skip))
    print(summarize_stages(timelines, args.skip, args.top_stages))


if __name__ == "__main__":
    main()
