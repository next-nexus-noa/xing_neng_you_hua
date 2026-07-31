# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

from vllm.v1.worker.compute_aware_ubatch import (
    ComputeAwareMicroBatch,
    RequestTokenSegment,
    _build_permutations,
    _build_request_cost_prefixes,
    _request_global_starts,
    _segment_cost,
    _segments_for_global_interval,
    _uniform_capacities,
)


DEFAULT_SCOM_SHAPE_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_SCOM_CAPACITY_QUANTUM = 64
DEFAULT_SCOM_MAX_CAPACITY_CANDIDATES = 8
DEFAULT_SCOM_MAX_SWAPS = 4
SCOM_SHAPE_OVERHEAD_WEIGHT = 0.12
SCOM_RANK0_PREFILL_WEIGHT = 0.08
SCOM_RANK1_DECODE_WEIGHT = 0.05
SCOM_RANK1_CONTEXT_WEIGHT = 0.04
SCOM_BROADCAST_STARTUP_COST = 4.0
SCOM_BROADCAST_TOKEN_WEIGHT = 0.035


@dataclass(frozen=True)
class ScomStageCost:
    rank0: float
    rank1: float
    broadcast: float
    padded_tokens: int
    prefill_tokens: int
    decode_tokens: int
    context_ratio: float

    def to_payload(self) -> dict[str, float | int]:
        return {
            "rank0": self.rank0,
            "rank1": self.rank1,
            "broadcast": self.broadcast,
            "padded_tokens": self.padded_tokens,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "context_ratio": self.context_ratio,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> ScomStageCost:
        return cls(
            rank0=float(payload["rank0"]),
            rank1=float(payload["rank1"]),
            broadcast=float(payload["broadcast"]),
            padded_tokens=int(payload["padded_tokens"]),
            prefill_tokens=int(payload["prefill_tokens"]),
            decode_tokens=int(payload["decode_tokens"]),
            context_ratio=float(payload["context_ratio"]),
        )


@dataclass(frozen=True)
class ScomGroupingPlan:
    groups: tuple[ComputeAwareMicroBatch, ...]
    permutation: tuple[int, ...]
    inverse_permutation: tuple[int, ...]
    baseline_capacities: tuple[int, ...]
    selected_capacities: tuple[int, ...]
    selected_shape_buckets: tuple[int, ...]
    baseline_objective: float
    selected_objective: float
    stage_costs: tuple[ScomStageCost, ...]
    predicted_gain: float
    applied: bool
    reason: str
    decision_overhead_us: float
    capacity_candidate_count: int
    shape_bucket_rejection_count: int
    mapping_swap_count: int
    selection_source: str
    cache_hit: bool = False
    cost_model: str = "analytical_shape_aware_with_compute_guardrail"

    @property
    def num_ubatches(self) -> int:
        return len(self.groups)

    def to_payload(self) -> dict[str, Any]:
        return {
            "groups": [group.to_payload() for group in self.groups],
            "baseline_capacities": list(self.baseline_capacities),
            "selected_capacities": list(self.selected_capacities),
            "selected_shape_buckets": list(
                self.selected_shape_buckets
            ),
            "baseline_objective": self.baseline_objective,
            "selected_objective": self.selected_objective,
            "stage_costs": [
                stage.to_payload() for stage in self.stage_costs
            ],
            "predicted_gain": self.predicted_gain,
            "applied": self.applied,
            "reason": self.reason,
            "decision_overhead_us": self.decision_overhead_us,
            "capacity_candidate_count": self.capacity_candidate_count,
            "shape_bucket_rejection_count": (
                self.shape_bucket_rejection_count
            ),
            "mapping_swap_count": self.mapping_swap_count,
            "selection_source": self.selection_source,
            "cache_hit": self.cache_hit,
            "cost_model": self.cost_model,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> ScomGroupingPlan:
        groups = tuple(
            ComputeAwareMicroBatch.from_payload(item)
            for item in payload["groups"]
        )
        total_tokens = sum(group.num_tokens for group in groups)
        permutation, inverse_permutation = _build_permutations(
            groups,
            total_tokens,
        )
        return cls(
            groups=groups,
            permutation=permutation,
            inverse_permutation=inverse_permutation,
            baseline_capacities=tuple(
                int(value) for value in payload["baseline_capacities"]
            ),
            selected_capacities=tuple(
                int(value) for value in payload["selected_capacities"]
            ),
            selected_shape_buckets=tuple(
                int(value)
                for value in payload["selected_shape_buckets"]
            ),
            baseline_objective=float(payload["baseline_objective"]),
            selected_objective=float(payload["selected_objective"]),
            stage_costs=tuple(
                ScomStageCost.from_payload(item)
                for item in payload["stage_costs"]
            ),
            predicted_gain=float(payload["predicted_gain"]),
            applied=bool(payload["applied"]),
            reason=str(payload["reason"]),
            decision_overhead_us=float(
                payload["decision_overhead_us"]
            ),
            capacity_candidate_count=int(
                payload["capacity_candidate_count"]
            ),
            shape_bucket_rejection_count=int(
                payload["shape_bucket_rejection_count"]
            ),
            mapping_swap_count=int(payload["mapping_swap_count"]),
            selection_source=str(
                payload.get(
                    "selection_source",
                    "uniform",
                )
            ),
            cache_hit=bool(payload.get("cache_hit", False)),
            cost_model=str(
                payload.get(
                    "cost_model",
                    "analytical_shape_aware_with_compute_guardrail",
                )
            ),
        )


def parse_shape_buckets(spec: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(spec, str):
        values = [
            int(value.strip())
            for value in spec.split(",")
            if value.strip()
        ]
    else:
        values = [int(value) for value in spec]
    if not values:
        raise ValueError("SCOM shape buckets must not be empty.")
    if any(value <= 0 for value in values):
        raise ValueError("SCOM shape buckets must be positive.")
    if values != sorted(set(values)):
        raise ValueError(
            "SCOM shape buckets must be strictly increasing."
        )
    return tuple(values)


def shape_bucket_for_tokens(
    num_tokens: int,
    shape_buckets: Sequence[int],
) -> int:
    if num_tokens < 0:
        raise ValueError("Token count must be non-negative.")
    for bucket in shape_buckets:
        if num_tokens <= int(bucket):
            return int(bucket)
    return num_tokens


def generate_bucket_safe_capacities(
    *,
    total_tokens: int,
    num_ubatches: int,
    quantum: int,
    shape_buckets: Sequence[int] = DEFAULT_SCOM_SHAPE_BUCKETS,
    max_candidates: int = DEFAULT_SCOM_MAX_CAPACITY_CANDIDATES,
    optimize_capacities: bool = True,
    allow_bucket_crossing: bool = False,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    if quantum < 1:
        raise ValueError("SCOM capacity quantum must be positive.")
    if max_candidates < 1:
        raise ValueError(
            "SCOM max capacity candidates must be positive."
        )
    buckets = parse_shape_buckets(shape_buckets)
    baseline = tuple(_uniform_capacities(total_tokens, num_ubatches))
    if not optimize_capacities or num_ubatches <= 1:
        return (baseline,), 0

    baseline_bucket = shape_bucket_for_tokens(
        max(baseline),
        buckets,
    )
    accepted: list[tuple[int, ...]] = [baseline]
    seen = {baseline}
    queue: deque[tuple[int, ...]] = deque([baseline])
    rejected = 0

    while queue and len(accepted) < max_candidates:
        current = queue.popleft()
        for slot in range(num_ubatches - 1):
            for direction in (-1, 1):
                candidate = list(current)
                src = slot if direction > 0 else slot + 1
                dst = slot + 1 if direction > 0 else slot
                if candidate[src] <= quantum:
                    continue
                transfer = quantum
                candidate[src] -= transfer
                candidate[dst] += transfer
                candidate_tuple = tuple(candidate)
                if candidate_tuple in seen:
                    continue
                seen.add(candidate_tuple)
                crosses_bucket = any(
                    shape_bucket_for_tokens(value, buckets)
                    > baseline_bucket
                    for value in candidate_tuple
                )
                if crosses_bucket and not allow_bucket_crossing:
                    rejected += 1
                    continue
                accepted.append(candidate_tuple)
                queue.append(candidate_tuple)
                if len(accepted) >= max_candidates:
                    break
            if len(accepted) >= max_candidates:
                break

    return tuple(accepted), rejected


def _contiguous_groups(
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


def _cost_balanced_groups(
    *,
    capacities: Sequence[int],
    scheduled: Sequence[int],
    cost_prefixes: Sequence[Sequence[float]],
    global_starts: Sequence[int],
    quantum: int,
) -> tuple[ComputeAwareMicroBatch, ...]:
    offsets = [0] * len(scheduled)
    remaining_cost = sum(prefix[-1] for prefix in cost_prefixes)
    groups: list[ComputeAwareMicroBatch] = []

    for group_index, capacity in enumerate(capacities):
        starts = offsets.copy()
        group_cost = 0.0
        remaining_capacity = int(capacity)
        if group_index == len(capacities) - 1:
            offsets = list(scheduled)
        else:
            target_cost = remaining_cost / (
                len(capacities) - group_index
            )
            while remaining_capacity > 0:
                desired_cost_per_token = (
                    target_cost - group_cost
                ) / remaining_capacity
                choices: list[
                    tuple[float, float, int, int, float]
                ] = []
                for request_index, request_tokens in enumerate(
                    scheduled
                ):
                    request_remaining = (
                        int(request_tokens) - offsets[request_index]
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
                    choices.append(
                        (
                            abs(
                                block_cost / take
                                - desired_cost_per_token
                            ),
                            -block_cost,
                            request_index,
                            take,
                            block_cost,
                        )
                    )
                if not choices:
                    raise ValueError(
                        "Unable to fill SCOM micro-batch."
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

        if group_index == len(capacities) - 1:
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
        if group.num_tokens != int(capacity):
            raise ValueError(
                "SCOM group does not match its token capacity."
            )
        groups.append(group)
        remaining_cost -= group_cost

    return tuple(groups)


def _stage_costs(
    *,
    groups: Sequence[ComputeAwareMicroBatch],
    scheduled: Sequence[int],
    computed: Sequence[int],
    shape_buckets: Sequence[int],
) -> tuple[ScomStageCost, ...]:
    maximum_position = max(
        (
            int(scheduled_tokens) + int(computed_tokens)
            for scheduled_tokens, computed_tokens in zip(
                scheduled,
                computed,
            )
            if int(scheduled_tokens) > 0
        ),
        default=1,
    )
    costs: list[ScomStageCost] = []
    for group in groups:
        prefill_tokens = 0
        decode_tokens = 0
        context_sum = 0.0
        for segment in group.segments:
            segment_tokens = segment.num_tokens
            if int(scheduled[segment.request_index]) > 1:
                prefill_tokens += segment_tokens
            else:
                decode_tokens += segment_tokens
            first_position = (
                int(computed[segment.request_index])
                + segment.request_token_start
                + 1
            )
            last_position = (
                int(computed[segment.request_index])
                + segment.request_token_stop
            )
            context_sum += (
                (first_position + last_position)
                * segment_tokens
                / 2.0
            )

        num_tokens = max(group.num_tokens, 1)
        context_ratio = min(
            1.0,
            context_sum / (num_tokens * maximum_position),
        )
        prefill_ratio = prefill_tokens / num_tokens
        decode_ratio = decode_tokens / num_tokens
        padded_tokens = shape_bucket_for_tokens(
            group.num_tokens,
            shape_buckets,
        )
        shape_overhead = (
            SCOM_SHAPE_OVERHEAD_WEIGHT * padded_tokens
        )
        rank0 = (
            group.predicted_cost
            * (
                1.0
                + SCOM_RANK0_PREFILL_WEIGHT * prefill_ratio
            )
            + shape_overhead
        )
        rank1 = (
            group.predicted_cost
            * (
                1.0
                + SCOM_RANK1_DECODE_WEIGHT * decode_ratio
                + SCOM_RANK1_CONTEXT_WEIGHT * context_ratio
            )
            + shape_overhead
        )
        broadcast = (
            SCOM_BROADCAST_STARTUP_COST
            + SCOM_BROADCAST_TOKEN_WEIGHT * padded_tokens
        )
        costs.append(
            ScomStageCost(
                rank0=rank0,
                rank1=rank1,
                broadcast=broadcast,
                padded_tokens=padded_tokens,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                context_ratio=context_ratio,
            )
        )
    return tuple(costs)


def _pipeline_objective(
    stage_costs: Sequence[ScomStageCost],
) -> float:
    rank0_finish = 0.0
    rank1_finish = 0.0
    broadcast_finish = 0.0
    for cost in stage_costs:
        rank0_finish += cost.rank0
        broadcast_start = max(rank0_finish, broadcast_finish)
        broadcast_finish = broadcast_start + cost.broadcast
        rank1_start = max(rank1_finish, broadcast_finish)
        rank1_finish = rank1_start + cost.rank1
    return rank1_finish


def _groups_are_causally_valid(
    groups: Sequence[ComputeAwareMicroBatch],
    num_requests: int,
) -> bool:
    previous_stops = [0] * num_requests
    for group in groups:
        request_indices = [
            segment.request_index for segment in group.segments
        ]
        if len(request_indices) != len(set(request_indices)):
            return False
        for segment in group.segments:
            if (
                segment.request_token_start
                != previous_stops[segment.request_index]
            ):
                return False
            previous_stops[segment.request_index] = (
                segment.request_token_stop
            )
    return True


def _try_adjacent_segment_swaps(
    *,
    groups: tuple[ComputeAwareMicroBatch, ...],
    scheduled: Sequence[int],
    computed: Sequence[int],
    cost_prefixes: Sequence[Sequence[float]],
    shape_buckets: Sequence[int],
    max_swaps: int,
) -> tuple[
    tuple[ComputeAwareMicroBatch, ...],
    tuple[ScomStageCost, ...],
    float,
    int,
]:
    best_groups = groups
    best_stage_costs = _stage_costs(
        groups=groups,
        scheduled=scheduled,
        computed=computed,
        shape_buckets=shape_buckets,
    )
    best_objective = _pipeline_objective(best_stage_costs)
    accepted_swaps = 0

    for _ in range(max_swaps):
        improved = False
        for group_index in range(len(best_groups) - 1):
            left = best_groups[group_index]
            right = best_groups[group_index + 1]
            for left_index, left_segment in enumerate(left.segments):
                for right_index, right_segment in enumerate(
                    right.segments
                ):
                    if (
                        left_segment.num_tokens
                        != right_segment.num_tokens
                        or left_segment.request_index
                        == right_segment.request_index
                    ):
                        continue
                    trial = [list(group.segments) for group in best_groups]
                    trial[group_index][left_index] = right_segment
                    trial[group_index + 1][right_index] = left_segment
                    trial_groups: list[ComputeAwareMicroBatch] = []
                    for segments in trial:
                        ordered = tuple(
                            sorted(
                                segments,
                                key=lambda segment: (
                                    segment.request_index,
                                    segment.request_token_start,
                                ),
                            )
                        )
                        trial_groups.append(
                            ComputeAwareMicroBatch(
                                segments=ordered,
                                predicted_cost=sum(
                                    _segment_cost(
                                        cost_prefixes[
                                            segment.request_index
                                        ],
                                        segment.request_token_start,
                                        segment.request_token_stop,
                                    )
                                    for segment in ordered
                                ),
                            )
                        )
                    trial_tuple = tuple(trial_groups)
                    if not _groups_are_causally_valid(
                        trial_tuple,
                        len(scheduled),
                    ):
                        continue
                    trial_costs = _stage_costs(
                        groups=trial_tuple,
                        scheduled=scheduled,
                        computed=computed,
                        shape_buckets=shape_buckets,
                    )
                    objective = _pipeline_objective(trial_costs)
                    if objective + 1e-9 >= best_objective:
                        continue
                    best_groups = trial_tuple
                    best_stage_costs = trial_costs
                    best_objective = objective
                    accepted_swaps += 1
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return (
        best_groups,
        best_stage_costs,
        best_objective,
        accepted_swaps,
    )


def plan_scom_groups(
    *,
    num_scheduled_tokens: Sequence[int],
    num_computed_tokens: Sequence[int],
    num_ubatches: int,
    quantum: int = 8,
    capacity_quantum: int = DEFAULT_SCOM_CAPACITY_QUANTUM,
    min_predicted_gain: float = 0.03,
    shape_buckets: Sequence[int] = DEFAULT_SCOM_SHAPE_BUCKETS,
    optimize_capacities: bool = True,
    allow_bucket_crossing: bool = False,
    max_capacity_candidates: int = (
        DEFAULT_SCOM_MAX_CAPACITY_CANDIDATES
    ),
    max_swaps: int = DEFAULT_SCOM_MAX_SWAPS,
) -> ScomGroupingPlan:
    start_ns = time.perf_counter_ns()
    scheduled = [int(value) for value in num_scheduled_tokens]
    computed = [int(value) for value in num_computed_tokens]
    if len(scheduled) != len(computed):
        raise ValueError(
            "Scheduled-token and computed-token arrays must have "
            "equal size."
        )
    if any(value < 0 for value in scheduled + computed):
        raise ValueError("Token counts must be non-negative.")
    if quantum < 1:
        raise ValueError("SCOM grouping quantum must be positive.")
    if max_swaps < 0:
        raise ValueError("SCOM max swaps must be non-negative.")
    if not 0.0 <= min_predicted_gain <= 1.0:
        raise ValueError(
            "SCOM minimum predicted gain must be between 0 and 1."
        )

    total_tokens = sum(scheduled)
    buckets = parse_shape_buckets(shape_buckets)
    baseline_capacities = tuple(
        _uniform_capacities(total_tokens, num_ubatches)
    )
    global_starts = _request_global_starts(scheduled)
    cost_prefixes = _build_request_cost_prefixes(
        scheduled,
        computed,
    )
    flat_costs = [
        _segment_cost(prefix, offset, offset + 1)
        for prefix in cost_prefixes
        for offset in range(len(prefix) - 1)
    ]
    baseline_groups = _contiguous_groups(
        capacities=baseline_capacities,
        scheduled=scheduled,
        global_starts=global_starts,
        flat_costs=flat_costs,
    )
    baseline_stage_costs = _stage_costs(
        groups=baseline_groups,
        scheduled=scheduled,
        computed=computed,
        shape_buckets=buckets,
    )
    baseline_objective = _pipeline_objective(
        baseline_stage_costs
    )
    capacity_candidates, bucket_rejections = (
        generate_bucket_safe_capacities(
            total_tokens=total_tokens,
            num_ubatches=num_ubatches,
            quantum=capacity_quantum,
            shape_buckets=buckets,
            max_candidates=max_capacity_candidates,
            optimize_capacities=optimize_capacities,
            allow_bucket_crossing=allow_bucket_crossing,
        )
    )

    # Preserve the already validated compute-aware composition candidate as a
    # guardrail.  The first SCOM implementation scored this candidate only
    # with an uncalibrated end-to-end analytical objective.  On real NPU
    # traces that rejected every candidate, even when the scalar
    # compute-aware model had previously produced a measured improvement.
    # Evaluate the equal-capacity composition once and retain it whenever its
    # critical-group reduction clears the configured gain threshold.
    guardrail_groups = _cost_balanced_groups(
        capacities=baseline_capacities,
        scheduled=scheduled,
        cost_prefixes=cost_prefixes,
        global_starts=global_starts,
        quantum=quantum,
    )
    guardrail_permutation, guardrail_inverse = _build_permutations(
        guardrail_groups,
        total_tokens,
    )
    uniform_critical_cost = max(
        group.predicted_cost for group in baseline_groups
    )
    guardrail_critical_cost = max(
        group.predicted_cost for group in guardrail_groups
    )
    guardrail_gain = (
        uniform_critical_cost - guardrail_critical_cost
    ) / max(uniform_critical_cost, 1e-9)
    guardrail_changed = (
        guardrail_permutation != tuple(range(total_tokens))
    )
    guardrail_applied = (
        guardrail_changed
        and guardrail_gain >= min_predicted_gain
    )

    if guardrail_applied:
        selected_groups = guardrail_groups
        selected_capacities = baseline_capacities
        selected_stage_costs = _stage_costs(
            groups=guardrail_groups,
            scheduled=scheduled,
            computed=computed,
            shape_buckets=buckets,
        )
        selected_objective = _pipeline_objective(
            selected_stage_costs
        )
        permutation = guardrail_permutation
        inverse_permutation = guardrail_inverse
        predicted_gain = guardrail_gain
        reason = "compute_aware_guardrail"
        selection_source = "compute_aware_guardrail"
        applied = True
    else:
        selected_groups = baseline_groups
        selected_capacities = baseline_capacities
        selected_stage_costs = baseline_stage_costs
        selected_objective = baseline_objective
        permutation, inverse_permutation = _build_permutations(
            baseline_groups,
            total_tokens,
        )
        predicted_gain = max(guardrail_gain, 0.0)
        reason = (
            "already_uniform_order"
            if not guardrail_changed
            else "predicted_gain_below_threshold"
        )
        selection_source = "uniform"
        applied = False

    best_swap_count = 0

    # The baseline capacity has already been evaluated by the guardrail.
    # Search only genuinely different, bucket-safe capacities.  This removes
    # the redundant swap search that dominated rejected SCOM decisions.
    for capacities in capacity_candidates[1:]:
        candidate_groups = _cost_balanced_groups(
            capacities=capacities,
            scheduled=scheduled,
            cost_prefixes=cost_prefixes,
            global_starts=global_starts,
            quantum=quantum,
        )
        (
            candidate_groups,
            candidate_stage_costs,
            candidate_objective,
            swap_count,
        ) = _try_adjacent_segment_swaps(
            groups=candidate_groups,
            scheduled=scheduled,
            computed=computed,
            cost_prefixes=cost_prefixes,
            shape_buckets=buckets,
            max_swaps=max_swaps,
        )
        candidate_gain = (
            baseline_objective - candidate_objective
        ) / max(baseline_objective, 1e-9)
        if (
            candidate_gain >= min_predicted_gain
            and candidate_gain > predicted_gain + 1e-9
        ):
            selected_groups = candidate_groups
            selected_capacities = capacities
            selected_stage_costs = candidate_stage_costs
            selected_objective = candidate_objective
            permutation, inverse_permutation = _build_permutations(
                candidate_groups,
                total_tokens,
            )
            predicted_gain = candidate_gain
            reason = "predicted_pipeline_makespan_reduced"
            selection_source = "shape_safe_pipeline_search"
            applied = True
            best_swap_count = swap_count

    return ScomGroupingPlan(
        groups=selected_groups,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        baseline_capacities=baseline_capacities,
        selected_capacities=selected_capacities,
        selected_shape_buckets=tuple(
            shape_bucket_for_tokens(capacity, buckets)
            for capacity in selected_capacities
        ),
        baseline_objective=baseline_objective,
        selected_objective=selected_objective,
        stage_costs=selected_stage_costs,
        predicted_gain=predicted_gain,
        applied=applied,
        reason=reason,
        decision_overhead_us=(
            time.perf_counter_ns() - start_ns
        ) / 1000.0,
        capacity_candidate_count=len(capacity_candidates),
        shape_bucket_rejection_count=bucket_rejections,
        mapping_swap_count=best_swap_count,
        selection_source=selection_source,
    )
