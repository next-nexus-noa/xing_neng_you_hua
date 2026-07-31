#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

import json
import os
import time
from contextlib import contextmanager
from typing import Any

import torch

from vllm.distributed.parallel_state import get_pp_group


def pp_trace_enabled() -> bool:
    return bool(int(os.getenv("VLLM_ASCEND_PP_TRACE", "0")))


def pp_trace_sync_enabled() -> bool:
    return bool(int(os.getenv("VLLM_ASCEND_PP_TRACE_SYNC", "0")))


def pp_trace_npu_timing_enabled() -> bool:
    return bool(int(os.getenv("VLLM_ASCEND_PP_TRACE_NPU_TIMING", "0")))


class PPTraceRecorder:
    """Per-step PP timeline recorder.

    The recorder is intentionally environment-variable driven so normal serving
    does not pay synchronization or file I/O overhead. Device synchronization is
    opt-in because it changes the asynchronous PP execution being observed.
    """

    def __init__(self, rank: int, pp_rank: int, pp_size: int) -> None:
        self.rank = rank
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.enabled = pp_trace_enabled()
        self.sync_enabled = pp_trace_sync_enabled()
        self.npu_timing_enabled = pp_trace_npu_timing_enabled()
        self._step_id = 0
        self._events: list[dict[str, Any]] = []
        self._step_meta: dict[str, Any] = {}
        self._path = os.getenv(
            "VLLM_ASCEND_PP_TRACE_FILE", "/tmp/vllm_ascend_pp_trace.jsonl"
        )

    def begin_step(self, total_tokens: int, num_reqs: int | None = None) -> None:
        if not self.enabled:
            return
        self._events = []
        self._step_meta = {
            "total_tokens": int(total_tokens),
            "num_reqs": None if num_reqs is None else int(num_reqs),
        }

    def update_step_meta(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        self._step_meta.update(kwargs)

    @contextmanager
    def stage(self, name: str, sync_device: bool = False):
        if not self.enabled:
            yield
            return
        do_sync = sync_device and self.sync_enabled and torch.npu.is_available()
        do_npu_timing = (
            self.npu_timing_enabled
            and sync_device
            and _is_forward_stage(name)
            and torch.npu.is_available()
        )
        if do_sync:
            torch.npu.synchronize()
        npu_start = torch.npu.Event(enable_timing=True) if do_npu_timing else None
        npu_end = torch.npu.Event(enable_timing=True) if do_npu_timing else None
        if npu_start is not None:
            npu_start.record()
        start = time.perf_counter()
        try:
            yield
        finally:
            if npu_end is not None:
                npu_end.record()
            if do_sync:
                torch.npu.synchronize()
            end = time.perf_counter()
            event = {
                "stage": name,
                "start_ms": start * 1000.0,
                "end_ms": end * 1000.0,
                "duration_ms": (end - start) * 1000.0,
            }
            if npu_start is not None and npu_end is not None:
                event["_npu_start_event"] = npu_start
                event["_npu_end_event"] = npu_end
            self._events.append(event)

    def record_event(self, name: str, start: float, end: float) -> None:
        if not self.enabled:
            return
        self._events.append(
            {
                "stage": name,
                "start_ms": start * 1000.0,
                "end_ms": end * 1000.0,
                "duration_ms": (end - start) * 1000.0,
            }
        )

    def finish_step(self) -> None:
        if not self.enabled:
            return
        self._finalize_npu_events()
        record = {
            "type": "rank_timeline",
            "step": self._step_id,
            "rank": self.rank,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            **self._step_meta,
            "events": self._events,
        }
        self._write(record)
        if self.pp_size == 2:
            self._write_dual_pp_metrics(record)
        self._step_id += 1

    def _finalize_npu_events(self) -> None:
        pending = [
            e for e in self._events
            if "_npu_start_event" in e and "_npu_end_event" in e
        ]
        if not pending:
            return
        torch.npu.synchronize()
        for event in pending:
            start_event = event.pop("_npu_start_event")
            end_event = event.pop("_npu_end_event")
            event["npu_duration_ms"] = start_event.elapsed_time(end_event)

    def _write_dual_pp_metrics(self, local_record: dict[str, Any]) -> None:
        pp = get_pp_group()
        gathered: list[Any] = [None for _ in range(pp.world_size)]
        torch.distributed.all_gather_object(
            gathered, local_record, group=pp.cpu_group
        )
        if self.pp_rank != 0:
            return
        metrics = _compute_dual_pp_metrics(gathered)
        if metrics is not None:
            self._write(metrics)

    def _write(self, record: dict[str, Any]) -> None:
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _stage_events(record: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    return [
        e for e in record["events"]
        if e["stage"] == stage or e["stage"].startswith(stage + ".")
    ]


def _is_forward_stage(stage: str) -> bool:
    return stage == "runner.forward" or stage.startswith("runner.forward.")


def _stage_total(record: dict[str, Any], prefix: str) -> float:
    return sum(
        e["duration_ms"]
        for e in record["events"]
        if e["stage"] == prefix or e["stage"].startswith(prefix + ".")
    )


def _union_intervals(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
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


def _intervals_total_ms(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _union_ms(events: list[dict[str, Any]]) -> float:
    return _intervals_total_ms(_union_intervals(events))


def _interval_intersection_ms(
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


def _compute_dual_pp_metrics(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(records) != 2 or any(r is None for r in records):
        return None
    records = sorted(records, key=lambda r: r["pp_rank"])
    step_start = min(e["start_ms"] for r in records for e in r["events"])
    step_end = max(e["end_ms"] for r in records for e in r["events"])
    makespan = max(0.0, step_end - step_start)
    if makespan == 0.0:
        return None

    compute_intervals = [
        _union_intervals(_stage_events(r, "runner.forward")) for r in records
    ]
    compute_ms = [_intervals_total_ms(intervals) for intervals in compute_intervals]
    compute_sum_ms = [_stage_total(r, "runner.forward") for r in records]
    active_ms = [_union_ms(r["events"]) for r in records]
    overlap = _interval_intersection_ms(compute_intervals[0], compute_intervals[1])
    total_compute = sum(compute_ms)
    bubble_ms = max(0.0, 2 * makespan - total_compute)
    metrics = {
        "type": "dual_pp_metrics",
        "step": records[0]["step"],
        "pp_size": 2,
        "total_tokens": records[0].get("total_tokens"),
        "makespan_ms": makespan,
        "rank_active_ms": active_ms,
        "rank_compute_ms": compute_ms,
        "rank_compute_sum_ms": compute_sum_ms,
        "rank_compute_utilization": [v / makespan for v in compute_ms],
        "rank_active_utilization": [v / makespan for v in active_ms],
        "bubble_ms": bubble_ms,
        "bubble_ratio": bubble_ms / (2 * makespan),
        "compute_overlap_ms": overlap,
        "compute_overlap_ratio": (
            overlap / min(compute_ms) if min(compute_ms) > 0 else 0.0
        ),
    }
    npu_compute_ms = [
        sum(
            float(e["npu_duration_ms"])
            for e in _stage_events(r, "runner.forward")
            if "npu_duration_ms" in e
        )
        for r in records
    ]
    if any(v > 0 for v in npu_compute_ms):
        metrics["rank_npu_compute_ms"] = npu_compute_ms
    return metrics
