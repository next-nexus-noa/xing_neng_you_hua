# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.scom_ubatch import (
    ScomGroupingPlan,
    generate_bucket_safe_capacities,
    parse_shape_buckets,
    plan_scom_groups,
)


def test_saturated_m2_has_no_safe_nonuniform_capacity() -> None:
    candidates, rejected = generate_bucket_safe_capacities(
        total_tokens=2048,
        num_ubatches=2,
        quantum=8,
        shape_buckets=(128, 256, 512, 1024, 2048),
        max_candidates=32,
    )

    assert candidates == ((1024, 1024),)
    assert rejected > 0


def test_saturated_m4_has_no_safe_nonuniform_capacity() -> None:
    candidates, rejected = generate_bucket_safe_capacities(
        total_tokens=2048,
        num_ubatches=4,
        quantum=8,
        shape_buckets=(128, 256, 512, 1024, 2048),
        max_candidates=32,
    )

    assert candidates == ((512, 512, 512, 512),)
    assert rejected > 0


def test_saturated_plan_keeps_bucket_safe_uniform_capacities() -> None:
    plan = plan_scom_groups(
        num_scheduled_tokens=[1024, 1024],
        num_computed_tokens=[3072, 0],
        num_ubatches=2,
        min_predicted_gain=0.0,
    )

    assert plan.baseline_capacities == (1024, 1024)
    assert plan.selected_capacities == (1024, 1024)
    assert plan.capacity_candidate_count == 1
    assert plan.selected_shape_buckets == (1024, 1024)


def test_saturated_plan_preserves_compute_aware_composition_gain() -> None:
    plan = plan_scom_groups(
        num_scheduled_tokens=[1024, 1024],
        num_computed_tokens=[3072, 0],
        num_ubatches=2,
        min_predicted_gain=0.02,
    )

    assert plan.applied
    assert plan.reason == "compute_aware_guardrail"
    assert plan.selection_source == "compute_aware_guardrail"
    assert plan.selected_capacities == (1024, 1024)
    assert plan.permutation != tuple(range(2048))
    assert plan.predicted_gain >= 0.02


def test_single_request_uses_fast_uniform_fallback() -> None:
    plan = plan_scom_groups(
        num_scheduled_tokens=[2048],
        num_computed_tokens=[0],
        num_ubatches=2,
        min_predicted_gain=0.02,
    )

    assert not plan.applied
    assert plan.reason == "already_uniform_order"
    assert plan.selection_source == "uniform"
    assert plan.capacity_candidate_count == 1
    assert plan.mapping_swap_count == 0


def test_unsaturated_capacity_search_stays_in_same_bucket() -> None:
    candidates, _ = generate_bucket_safe_capacities(
        total_tokens=1600,
        num_ubatches=2,
        quantum=8,
        shape_buckets=(128, 256, 512, 1024, 2048),
        max_candidates=8,
    )

    assert candidates[0] == (800, 800)
    assert len(candidates) > 1
    assert any(left != right for left, right in candidates)
    assert all(max(candidate) <= 1024 for candidate in candidates)


def test_scom_plan_preserves_causality_and_inverse_permutation() -> None:
    scheduled = [600, 700, 300]
    plan = plan_scom_groups(
        num_scheduled_tokens=scheduled,
        num_computed_tokens=[2500, 500, 0],
        num_ubatches=2,
        quantum=8,
        min_predicted_gain=0.0,
        max_capacity_candidates=8,
    )

    previous_stops = [0] * len(scheduled)
    for group in plan.groups:
        assert group.num_tokens > 0
        request_indices = [
            segment.request_index for segment in group.segments
        ]
        assert len(request_indices) == len(set(request_indices))
        for segment in group.segments:
            request_index = segment.request_index
            assert (
                segment.request_token_start
                == previous_stops[request_index]
            )
            previous_stops[request_index] = (
                segment.request_token_stop
            )
    assert previous_stops == scheduled

    original = list(range(sum(scheduled)))
    permuted = [original[index] for index in plan.permutation]
    restored = [
        permuted[index] for index in plan.inverse_permutation
    ]
    assert restored == original


def test_scom_payload_round_trip() -> None:
    plan = plan_scom_groups(
        num_scheduled_tokens=[512, 512],
        num_computed_tokens=[2048, 0],
        num_ubatches=2,
        min_predicted_gain=0.0,
    )

    restored = ScomGroupingPlan.from_payload(plan.to_payload())
    assert restored == plan


def test_disabling_capacity_optimization_keeps_uniform_capacities() -> None:
    plan = plan_scom_groups(
        num_scheduled_tokens=[600, 700, 300],
        num_computed_tokens=[2500, 500, 0],
        num_ubatches=2,
        min_predicted_gain=0.0,
        optimize_capacities=False,
    )

    assert plan.capacity_candidate_count == 1
    assert plan.selected_capacities == (800, 800)


@pytest.mark.parametrize(
    "spec",
    ["", "128,128", "256,128", "0,128"],
)
def test_invalid_shape_buckets_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_shape_buckets(spec)
