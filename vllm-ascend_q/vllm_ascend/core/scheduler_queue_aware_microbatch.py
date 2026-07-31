#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Queue-aware PP micro-batch scheduler.

This scheduler is used for the M4 baseline only. It keeps vLLM's normal batch
construction untouched and chooses the PP micro-batch count solely from the
current scheduler queue length.
"""

from __future__ import annotations

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend import envs


def _parse_thresholds(raw: str) -> list[tuple[int, int]]:
    """Parse thresholds like ``0:1,4:2,16:4`` into sorted pairs."""
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            queue_s, micro_s = item.split(":", 1)
            queue_len = int(queue_s)
            microbatch_num = int(micro_s)
        except ValueError as exc:
            raise ValueError(
                "VLLM_ASCEND_PP_MICROBATCH_QUEUE_THRESHOLDS must use "
                "comma-separated queue_len:microbatch pairs, for example "
                "'0:1,4:2,16:4'."
            ) from exc
        if queue_len < 0:
            raise ValueError("Queue-aware micro-batch thresholds must be non-negative.")
        if microbatch_num < 1:
            raise ValueError("Queue-aware micro-batch counts must be >= 1.")
        pairs.append((queue_len, microbatch_num))
    if not pairs:
        raise ValueError("Queue-aware micro-batch thresholds cannot be empty.")
    pairs.sort(key=lambda pair: pair[0])
    if pairs[0][0] != 0:
        pairs.insert(0, (0, pairs[0][1]))
    return pairs


def _select_microbatch(queue_len: int, thresholds: list[tuple[int, int]]) -> int:
    selected = thresholds[0][1]
    for threshold, microbatch_num in thresholds:
        if queue_len < threshold:
            break
        selected = microbatch_num
    return selected


class QueueAwareMicrobatchScheduler(Scheduler):
    """Default scheduler plus queue-length-only PP micro-batch selection."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pp_microbatch_thresholds = _parse_thresholds(
            envs.VLLM_ASCEND_PP_MICROBATCH_QUEUE_THRESHOLDS
        )
        logger.info(
            "Queue-aware PP micro-batch enabled with thresholds: %s",
            self._pp_microbatch_thresholds,
        )

    def schedule(self):
        queue_len = len(self.waiting) + len(self.running)
        scheduler_output = super().schedule()
        selected_m = _select_microbatch(queue_len, self._pp_microbatch_thresholds)
        scheduler_output.pp_queue_len = queue_len
        scheduler_output.pp_microbatch_num = selected_m
        logger.debug(
            "Queue-aware PP micro-batch decision: queue_len=%d selected_m=%d",
            queue_len,
            selected_m,
        )
        return scheduler_output
