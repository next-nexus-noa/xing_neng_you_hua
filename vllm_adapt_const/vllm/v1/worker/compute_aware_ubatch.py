# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence


COMPUTE_AWARE_GROUPING_QUANTUM = 8
COMPUTE_AWARE_CONTEXT_WEIGHT = 1.0
COMPUTE_AWARE_MIN_PREDICTED_GAIN = 0.02


@dataclass(frozen=True)
class RequestTokenSegment:
    """A contiguous part of one request in the original flattened batch."""

    request_index: int
    request_token_start: int
    request_token_stop: int
    global_token_start: int
    global_token_stop: int

    @property
    def num_tokens(self) -> int:
        return self.request_token_stop - self.request_token_start

    def to_payload(self) -> dict[str, int]:
        return {
            "request_index": self.request_index,
            "request_token_start": self.request_token_start,
            "request_token_stop": self.request_token_stop,
            "global_token_start": self.global_token_start,
            "global_token_stop": self.global_token_stop,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> RequestTokenSegment:
        return cls(
            request_index=int(payload["request_index"]),
            request_token_start=int(payload["request_token_start"]),
            request_token_stop=int(payload["request_token_stop"]),
            global_token_start=int(payload["global_token_start"]),
            global_token_stop=int(payload["global_token_stop"]),
        )


@dataclass(frozen=True)
class ComputeAwareMicroBatch:
    segments: tuple[RequestTokenSegment, ...]
    predicted_cost: float

    @property
    def num_tokens(self) -> int:
        return sum(segment.num_tokens for segment in self.segments)

    @property
    def num_requests(self) -> int:
        return len(self.segments)

    @property
    def token_indices(self) -> tuple[int, ...]:
        return tuple(
            token_index
            for segment in self.segments
            for token_index in range(
                segment.global_token_start,
                segment.global_token_stop,
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "segments": [
                segment.to_payload() for segment in self.segments
            ],
            "predicted_cost": self.predicted_cost,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> ComputeAwareMicroBatch:
        return cls(
            segments=tuple(
                RequestTokenSegment.from_payload(item)
                for item in payload["segments"]
            ),
            predicted_cost=float(payload["predicted_cost"]),
        )


@dataclass(frozen=True)
class ComputeAwareGroupingPlan:
    groups: tuple[ComputeAwareMicroBatch, ...]
    permutation: tuple[int, ...]
    inverse_permutation: tuple[int, ...]
    uniform_group_costs: tuple[float, ...]
    candidate_group_costs: tuple[float, ...]
    predicted_gain: float
    applied: bool
    reason: str
    decision_overhead_us: float

    @property
    def num_ubatches(self) -> int:
        return len(self.groups)

    def to_payload(self) -> dict[str, Any]:
        return {
            "groups": [group.to_payload() for group in self.groups],
            "uniform_group_costs": list(self.uniform_group_costs),
            "candidate_group_costs": list(self.candidate_group_costs),
            "predicted_gain": self.predicted_gain,
            "applied": self.applied,
            "reason": self.reason,
            "decision_overhead_us": self.decision_overhead_us,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> ComputeAwareGroupingPlan:
        groups = tuple(
            ComputeAwareMicroBatch.from_payload(item)
            for item in payload["groups"]
        )
        permutation, inverse_permutation = _build_permutations(
            groups,
            sum(group.num_tokens for group in groups),
        )
        return cls(
            groups=groups,
            permutation=permutation,
            inverse_permutation=inverse_permutation,
            uniform_group_costs=tuple(
                float(item) for item in payload["uniform_group_costs"]
            ),
            candidate_group_costs=tuple(
                float(item) for item in payload["candidate_group_costs"]
            ),
            predicted_gain=float(payload["predicted_gain"]),
            applied=bool(payload["applied"]),
            reason=str(payload["reason"]),
            decision_overhead_us=float(
                payload["decision_overhead_us"]
            ),
        )


def _uniform_capacities(
    total_tokens: int,
    num_ubatches: int,
) -> list[int]:
    if num_ubatches < 1 or total_tokens < num_ubatches:
        raise ValueError(
            "Grouping requires total_tokens >= num_ubatches >= 1."
        )
    base, remainder = divmod(total_tokens, num_ubatches)
    return [base] * (num_ubatches - 1) + [base + remainder]


def should_plan_compute_aware_groups(
    *,
    total_tokens: int,
    min_tokens: int,
) -> bool:
    """Return whether a step is large enough to justify group planning."""
    if total_tokens < 0:
        raise ValueError("Total token count must be non-negative.")
    if min_tokens < 0:
        raise ValueError("Minimum token count must be non-negative.")
    return total_tokens >= min_tokens


def _request_global_starts(
    num_scheduled_tokens: Sequence[int],
) -> list[int]:
    starts: list[int] = []
    running = 0
    for value in num_scheduled_tokens:
        starts.append(running)
        running += int(value)
    return starts


def _build_request_cost_prefixes(
    num_scheduled_tokens: Sequence[int],
    num_computed_tokens: Sequence[int],
) -> list[list[float]]:
    maximum_position = max(
        (
            int(computed) + int(scheduled)
            for scheduled, computed in zip(
                num_scheduled_tokens,
                num_computed_tokens,
            )
            if int(scheduled) > 0
        ),
        default=1,
    )
    prefixes: list[list[float]] = []
    for scheduled, computed in zip(
        num_scheduled_tokens,
        num_computed_tokens,
    ):
        prefix = [0.0]
        for token_offset in range(int(scheduled)):
            position = int(computed) + token_offset + 1
            cost = (
                1.0
                + COMPUTE_AWARE_CONTEXT_WEIGHT
                * float(position)
                / maximum_position
            )
            prefix.append(prefix[-1] + cost)
        prefixes.append(prefix)
    return prefixes


def _segment_cost(
    cost_prefix: Sequence[float],
    start: int,
    stop: int,
) -> float:
    return float(cost_prefix[stop] - cost_prefix[start])


def _segments_for_global_interval(
    *,
    interval_start: int,
    interval_stop: int,
    scheduled: Sequence[int],
    global_starts: Sequence[int],
) -> tuple[RequestTokenSegment, ...]:
    segments: list[RequestTokenSegment] = []
    for request_index, (request_start, request_tokens) in enumerate(
        zip(global_starts, scheduled)
    ):
        request_stop = request_start + int(request_tokens)
        overlap_start = max(interval_start, request_start)
        overlap_stop = min(interval_stop, request_stop)
        if overlap_start >= overlap_stop:
            continue
        segments.append(
            RequestTokenSegment(
                request_index=request_index,
                request_token_start=overlap_start - request_start,
                request_token_stop=overlap_stop - request_start,
                global_token_start=overlap_start,
                global_token_stop=overlap_stop,
            )
        )
    return tuple(segments)


def _uniform_groups(
    *,
    capacities: Sequence[int],
    scheduled: Sequence[int],
    global_starts: Sequence[int],
    flat_costs: Sequence[float],
) -> tuple[ComputeAwareMicroBatch, ...]:
    groups: list[ComputeAwareMicroBatch] = []
    start = 0
    for capacity in capacities:
        stop = start + int(capacity)
        groups.append(
            ComputeAwareMicroBatch(
                segments=_segments_for_global_interval(
                    interval_start=start,
                    interval_stop=stop,
                    scheduled=scheduled,
                    global_starts=global_starts,
                ),
                predicted_cost=float(sum(flat_costs[start:stop])),
            )
        )
        start = stop
    return tuple(groups)


def _build_permutations(
    groups: Sequence[ComputeAwareMicroBatch],
    total_tokens: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    permutation = tuple(
        token_index
        for group in groups
        for token_index in group.token_indices
    )
    if len(permutation) != total_tokens:
        raise ValueError("Grouping plan does not cover every token.")
    if sorted(permutation) != list(range(total_tokens)):
        raise ValueError("Grouping plan is not a token permutation.")
    inverse = [0] * total_tokens
    for permuted_index, original_index in enumerate(permutation):
        inverse[original_index] = permuted_index
    return permutation, tuple(inverse)


def plan_compute_aware_groups(
    *,
    num_scheduled_tokens: Sequence[int],
    num_computed_tokens: Sequence[int],
    num_ubatches: int,
    quantum: int = COMPUTE_AWARE_GROUPING_QUANTUM,
    min_predicted_gain: float = COMPUTE_AWARE_MIN_PREDICTED_GAIN,
) -> ComputeAwareGroupingPlan:
    """Create equal-token, cost-balanced groups with a reversible permutation.

    Tokens from different requests may be interleaved across micro-batches.
    Within each request, however, token ranges are assigned to monotonically
    increasing micro-batch ids. This preserves the dependency that a later
    prefill range must not execute before its earlier KV range.
    """
    start_ns = time.perf_counter_ns()
    scheduled = [int(value) for value in num_scheduled_tokens]
    computed = [int(value) for value in num_computed_tokens]
    if len(scheduled) != len(computed):
        raise ValueError(
            "Scheduled-token and computed-token arrays must have equal size."
        )
    if any(value < 0 for value in scheduled + computed):
        raise ValueError("Token counts must be non-negative.")
    if quantum < 1:
        raise ValueError("Grouping quantum must be positive.")

    total_tokens = sum(scheduled)
    capacities = _uniform_capacities(total_tokens, num_ubatches)
    global_starts = _request_global_starts(scheduled)
    cost_prefixes = _build_request_cost_prefixes(scheduled, computed)
    flat_costs = [
        _segment_cost(prefix, token_offset, token_offset + 1)
        for prefix in cost_prefixes
        for token_offset in range(len(prefix) - 1)
    ]
    uniform_groups = _uniform_groups(
        capacities=capacities,
        scheduled=scheduled,
        global_starts=global_starts,
        flat_costs=flat_costs,
    )
    uniform_costs = tuple(
        group.predicted_cost for group in uniform_groups
    )

    offsets = [0] * len(scheduled)
    remaining_cost = sum(prefix[-1] for prefix in cost_prefixes)
    candidate_groups: list[ComputeAwareMicroBatch] = []
    for ubatch_index, capacity in enumerate(capacities):
        starts = offsets.copy()
        group_cost = 0.0
        remaining_capacity = int(capacity)
        if ubatch_index == num_ubatches - 1:
            for request_index, request_tokens in enumerate(scheduled):
                offsets[request_index] = request_tokens
        else:
            target_cost = remaining_cost / (
                num_ubatches - ubatch_index
            )
            while remaining_capacity > 0:
                required_cost_per_token = (
                    target_cost - group_cost
                ) / remaining_capacity
                choices: list[
                    tuple[float, float, int, int, float]
                ] = []
                for request_index, request_tokens in enumerate(scheduled):
                    request_remaining = (
                        request_tokens - offsets[request_index]
                    )
                    if request_remaining <= 0:
                        continue
                    take = min(
                        quantum,
                        request_remaining,
                        remaining_capacity,
                    )
                    block_cost = _segment_cost(
                        cost_prefixes[request_index],
                        offsets[request_index],
                        offsets[request_index] + take,
                    )
                    unit_cost_distance = abs(
                        block_cost / take
                        - required_cost_per_token
                    )
                    choices.append(
                        (
                            unit_cost_distance,
                            -block_cost,
                            request_index,
                            take,
                            block_cost,
                        )
                    )
                if not choices:
                    raise ValueError(
                        "Unable to fill compute-aware micro-batch."
                    )
                _, _, request_index, take, block_cost = min(choices)
                offsets[request_index] += take
                remaining_capacity -= take
                group_cost += block_cost

        segments: list[RequestTokenSegment] = []
        for request_index, (start, stop) in enumerate(
            zip(starts, offsets)
        ):
            if start == stop:
                continue
            global_start = global_starts[request_index] + start
            global_stop = global_starts[request_index] + stop
            segments.append(
                RequestTokenSegment(
                    request_index=request_index,
                    request_token_start=start,
                    request_token_stop=stop,
                    global_token_start=global_start,
                    global_token_stop=global_stop,
                )
            )
        if ubatch_index == num_ubatches - 1:
            group_cost = sum(
                _segment_cost(
                    cost_prefixes[segment.request_index],
                    segment.request_token_start,
                    segment.request_token_stop,
                )
                for segment in segments
            )
        group = ComputeAwareMicroBatch(
            segments=tuple(segments),
            predicted_cost=group_cost,
        )
        if group.num_tokens != capacity:
            raise ValueError(
                "Compute-aware group does not match its token capacity."
            )
        candidate_groups.append(group)
        remaining_cost -= group_cost

    candidate_costs = tuple(
        group.predicted_cost for group in candidate_groups
    )
    candidate_permutation, candidate_inverse = _build_permutations(
        candidate_groups,
        total_tokens,
    )
    uniform_critical_cost = max(uniform_costs)
    candidate_critical_cost = max(candidate_costs)
    predicted_gain = (
        uniform_critical_cost - candidate_critical_cost
    ) / max(uniform_critical_cost, 1e-9)
    reordered = candidate_permutation != tuple(range(total_tokens))
    applied = reordered and predicted_gain >= min_predicted_gain

    if applied:
        groups = tuple(candidate_groups)
        permutation = candidate_permutation
        inverse_permutation = candidate_inverse
        reason = "predicted_critical_path_reduced"
    else:
        groups = uniform_groups
        permutation, inverse_permutation = _build_permutations(
            uniform_groups,
            total_tokens,
        )
        reason = (
            "already_uniform_order"
            if not reordered
            else "predicted_gain_below_threshold"
        )

    return ComputeAwareGroupingPlan(
        groups=groups,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        uniform_group_costs=uniform_costs,
        candidate_group_costs=candidate_costs,
        predicted_gain=predicted_gain,
        applied=applied,
        reason=reason,
        decision_overhead_us=(
            time.perf_counter_ns() - start_ns
        ) / 1000.0,
    )
