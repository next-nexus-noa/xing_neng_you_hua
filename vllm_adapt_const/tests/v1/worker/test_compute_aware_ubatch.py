# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.compute_aware_ubatch import (
    ComputeAwareGroupingPlan,
    plan_compute_aware_groups,
    should_plan_compute_aware_groups,
)


def test_groups_balance_saturated_prefill_without_changing_shape() -> None:
    plan = plan_compute_aware_groups(
        num_scheduled_tokens=[1024, 1024],
        num_computed_tokens=[3072, 0],
        num_ubatches=2,
    )

    assert plan.applied
    assert [group.num_tokens for group in plan.groups] == [1024, 1024]
    assert plan.predicted_gain >= 0.02
    assert max(plan.candidate_group_costs) < max(
        plan.uniform_group_costs
    )
    assert plan.permutation != tuple(range(2048))


def test_request_tokens_keep_monotonic_microbatch_order() -> None:
    scheduled = [600, 700, 748]
    plan = plan_compute_aware_groups(
        num_scheduled_tokens=scheduled,
        num_computed_tokens=[2500, 500, 0],
        num_ubatches=4,
        min_predicted_gain=0.0,
    )

    previous_stops = [0] * len(scheduled)
    for group in plan.groups:
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


def test_inverse_permutation_restores_original_order() -> None:
    plan = plan_compute_aware_groups(
        num_scheduled_tokens=[8, 8, 8, 8],
        num_computed_tokens=[3000, 0, 2000, 100],
        num_ubatches=2,
        quantum=1,
        min_predicted_gain=0.0,
    )
    original = list(range(32))
    permuted = [original[index] for index in plan.permutation]
    restored = [
        permuted[index] for index in plan.inverse_permutation
    ]
    assert restored == original


def test_single_request_falls_back_to_contiguous_uniform_groups() -> None:
    plan = plan_compute_aware_groups(
        num_scheduled_tokens=[2048],
        num_computed_tokens=[0],
        num_ubatches=2,
    )

    assert not plan.applied
    assert plan.reason == "already_uniform_order"
    assert plan.permutation == tuple(range(2048))


def test_plan_payload_round_trip() -> None:
    plan = plan_compute_aware_groups(
        num_scheduled_tokens=[16, 16],
        num_computed_tokens=[1024, 0],
        num_ubatches=2,
        quantum=1,
        min_predicted_gain=0.0,
    )
    restored = ComputeAwareGroupingPlan.from_payload(
        plan.to_payload()
    )
    assert restored == plan


def test_compute_aware_minimum_token_gate() -> None:
    assert not should_plan_compute_aware_groups(
        total_tokens=511,
        min_tokens=512,
    )
    assert should_plan_compute_aware_groups(
        total_tokens=512,
        min_tokens=512,
    )


@pytest.mark.parametrize(
    ("total_tokens", "min_tokens"),
    [(-1, 0), (1, -1)],
)
def test_compute_aware_minimum_token_gate_rejects_negative_values(
    total_tokens: int,
    min_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        should_plan_compute_aware_groups(
            total_tokens=total_tokens,
            min_tokens=min_tokens,
        )
