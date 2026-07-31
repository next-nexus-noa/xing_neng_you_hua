# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.adaptive_ubatch import (
    AdaptiveUBatchDecision,
    AdaptiveUBatchController,
    extract_adaptive_ubatch_features,
)


def _parallel_config(**overrides):
    values = {
        "adaptive_ubatch_bad_threshold_pct": 10.0,
        "adaptive_ubatch_candidate_calibration_observations": 0,
        "adaptive_ubatch_cold_start_penalty_ratio": 0.0,
        "adaptive_ubatch_cooldown_steps": 8,
        "adaptive_ubatch_enable_queue_guard": False,
        "adaptive_ubatch_enable_exploration": False,
        "adaptive_ubatch_ewma_alpha": 1.0,
        "adaptive_ubatch_failure_cooldown_steps": 8,
        "adaptive_ubatch_max_calibration_scale": 8.0,
        "adaptive_ubatch_max_size": 4,
        "adaptive_ubatch_max_uncertainty_ratio": 0.5,
        "adaptive_ubatch_max_exploration_regret_pct": 5.0,
        "adaptive_ubatch_min_gain_pct": 0.0,
        "adaptive_ubatch_min_hold_steps": 0,
        "adaptive_ubatch_min_observations": 1,
        "adaptive_ubatch_min_prefill_ratio_m4": 0.85,
        "adaptive_ubatch_min_tokens_m2": 128,
        "adaptive_ubatch_min_tokens_m4": 512,
        "adaptive_ubatch_mode": "calibrated_risk_aware",
        "adaptive_ubatch_prefill_threshold_pct": 85.0,
        "adaptive_ubatch_risk_kappa": 0.0,
        "adaptive_ubatch_safe_m": 1,
        "adaptive_ubatch_exploration_interval_steps": 64,
        "adaptive_ubatch_exploration_stable_steps": 8,
        "adaptive_ubatch_switch_confirmations": 2,
        "adaptive_ubatch_switch_threshold_pct": 0.0,
        "adaptive_ubatch_trace_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _model_config(model_billions: int):
    return SimpleNamespace(
        model=f"test-{model_billions}B",
        served_model_name=None,
        hf_config=SimpleNamespace(
            _name_or_path=f"test-{model_billions}B",
            hidden_size=5120 if model_billions >= 14 else 3584,
            num_hidden_layers=48 if model_billions >= 14 else 32,
        ),
    )


def _candidate(decision, m: int):
    return next(score for score in decision.candidate_scores if score["m"] == m)


@pytest.mark.parametrize(
    ("model_billions", "expected_model_bucket"),
    ((3, "3b"), (7, "7b"), (14, "14b")),
)
def test_calibration_buckets_separate_supported_model_sizes(
    model_billions,
    expected_model_bucket,
):
    features = extract_adaptive_ubatch_features(
        model_config=_model_config(model_billions),
        num_scheduled_tokens=np.array([1024, 1024], dtype=np.int32),
    )

    assert features.bucket_key == (
        expected_model_bucket,
        "prefill",
        "large",
    )


@pytest.mark.parametrize(
    ("model_billions", "expected_model_bucket"),
    ((3, "medium"), (7, "medium"), (14, "large")),
)
def test_best_20260729_profile_restores_coarse_model_buckets(
    monkeypatch,
    model_billions,
    expected_model_bucket,
):
    monkeypatch.setenv(
        "ADAPTIVE_UBATCH_POLICY_PROFILE",
        "best_20260729",
    )
    features = extract_adaptive_ubatch_features(
        model_config=_model_config(model_billions),
        num_scheduled_tokens=np.array([1024, 1024], dtype=np.int32),
    )

    assert features.bucket_key == (
        expected_model_bucket,
        "prefill",
        "large",
    )


@pytest.mark.parametrize(
    ("total_tokens", "expected_bucket"),
    (
        (127, "small"),
        (128, "small"),
        (255, "small"),
        (256, "medium"),
        (1023, "medium"),
        (1024, "large"),
        (4096, "large"),
    ),
)
def test_calibration_buckets_follow_npu_shape_boundaries(
    total_tokens,
    expected_bucket,
):
    features = extract_adaptive_ubatch_features(
        model_config=_model_config(7),
        num_scheduled_tokens=np.array([total_tokens], dtype=np.int32),
    )

    assert features.bucket_key[2] == expected_bucket


def test_calibration_buckets_pool_per_request_query_shapes():
    short_requests = extract_adaptive_ubatch_features(
        model_config=_model_config(7),
        num_scheduled_tokens=np.array([1024, 1024], dtype=np.int32),
    )
    one_long_request = extract_adaptive_ubatch_features(
        model_config=_model_config(7),
        num_scheduled_tokens=np.array([2048], dtype=np.int32),
    )

    assert short_requests.bucket_key == one_long_request.bucket_key


def test_calibration_buckets_pool_prefill_decode_composition():
    mostly_prefill = extract_adaptive_ubatch_features(
        model_config=_model_config(7),
        num_scheduled_tokens=np.array([1024, 1024], dtype=np.int32),
    )
    prefill_with_many_decodes = extract_adaptive_ubatch_features(
        model_config=_model_config(7),
        num_scheduled_tokens=np.array(
            [1024, 992, *([1] * 32)],
            dtype=np.int32,
        ),
    )

    assert mostly_prefill.total_tokens == prefill_with_many_decodes.total_tokens
    assert mostly_prefill.bucket_key == prefill_with_many_decodes.bucket_key


def test_small_model_unprofitable_prefill_falls_back_to_safe_m():
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_candidate_calibration_observations=1,
            adaptive_ubatch_max_size=2,
            adaptive_ubatch_switch_confirmations=1,
        ),
        model_config=_model_config(3),
    )
    prefill = np.array([1024, 1024], dtype=np.int32)

    for _ in range(6):
        decision = controller.select(prefill)
        prior_ms = _candidate(
            decision,
            decision.num_ubatches,
        )["prior_ms"]
        controller.observe(
            decision,
            forward_ms=prior_ms * (
                2.0 if decision.num_ubatches == 2 else 1.0
            ),
        )

    fallback = controller.select(prefill)

    assert fallback.num_ubatches == 1
    assert _candidate(fallback, 2)["rejected"] is True
    assert _candidate(fallback, 2)["rejection_reason"] == "cooldown"


def test_queue_pressure_is_preserved_in_feature_snapshot():
    features = extract_adaptive_ubatch_features(
        model_config=_model_config(3),
        num_scheduled_tokens=np.array([1024, 1024], dtype=np.int32),
        waiting_count=12,
        running_count=8,
        oldest_wait_ms=350.0,
        pending_first_token_count=15,
        oldest_first_token_wait_ms=900.0,
        pending_prefill_tokens=12000,
    )

    assert features.waiting_count == 12
    assert features.running_count == 8
    assert features.oldest_wait_ms == pytest.approx(350.0)
    assert features.pending_first_token_count == 15
    assert features.oldest_first_token_wait_ms == pytest.approx(900.0)
    assert features.pending_prefill_tokens == 12000
    assert features.prefill_reqs == 2
    assert features.decode_reqs == 0


def test_mirrored_controller_can_disable_trace_ownership(tmp_path):
    trace_path = tmp_path / "adaptive.jsonl"
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_trace_path=str(trace_path),
        ),
        model_config=_model_config(7),
        trace_enabled=False,
    )

    controller.select(np.array([1024, 1024], dtype=np.int32))
    controller.close_trace()

    assert controller._trace_path is None
    assert not trace_path.exists()


def test_mirrored_controllers_remain_deterministic_with_shared_feedback():
    config = _parallel_config(
        adaptive_ubatch_candidate_calibration_observations=1,
        adaptive_ubatch_switch_confirmations=1,
    )
    first = AdaptiveUBatchController(
        parallel_config=config,
        model_config=_model_config(7),
    )
    second = AdaptiveUBatchController(
        parallel_config=config,
        model_config=_model_config(7),
        trace_enabled=False,
    )
    workloads = (
        np.array([1024, 1024], dtype=np.int32),
        np.array([1, 1, 1, 1], dtype=np.int32),
        np.array([512, 1, 512, 1], dtype=np.int32),
    )

    for step in range(12):
        workload = workloads[step % len(workloads)]
        first_decision = first.select(workload)
        second_decision = second.select(workload)

        assert first_decision.num_ubatches == second_decision.num_ubatches
        assert first_decision.reason == second_decision.reason
        assert first_decision.bucket_key == second_decision.bucket_key
        assert first_decision.switched == second_decision.switched

        measured_ms = 10.0 + step
        first.observe(first_decision, forward_ms=measured_ms)
        second.observe(second_decision, forward_ms=measured_ms)


def test_ineligible_safe_m_uses_multiplicative_calibration(tmp_path):
    trace_path = tmp_path / "adaptive.jsonl"
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_trace_path=str(trace_path),
        ),
        model_config=_model_config(14),
    )

    first = controller.select(np.array([1], dtype=np.int32))
    prior_ms = _candidate(first, 1)["prior_ms"]
    controller.observe(first, forward_ms=prior_ms * 6.0)
    second = controller.select(np.array([1], dtype=np.int32))
    calibrated_ms = _candidate(second, 1)["calibrated_ms"]

    assert calibrated_ms == pytest.approx(prior_ms * 6.0)
    assert _candidate(second, 1)["rejected"] is False

    controller.close_trace()
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["type"] for record in records] == [
        "adaptive_ubatch_ineligible",
        "adaptive_ubatch_observation",
        "adaptive_ubatch_ineligible",
    ]
    observation = records[1]
    assert observation["bad_execution"] is False
    assert observation["has_alternative"] is False
    assert observation["affects_cooldown"] is False


def test_switch_confirmation_and_bucket_state_are_isolated():
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(),
        model_config=_model_config(7),
    )
    prefill = np.array([1024, 1024], dtype=np.int32)

    first = controller.select(prefill)
    best_m = min(
        first.candidate_scores,
        key=lambda score: score["robust_ms"],
    )["m"]
    assert best_m > 1
    assert first.num_ubatches == 1
    assert first.reason == "switch_confirmation"

    second = controller.select(prefill)
    assert second.num_ubatches == best_m
    assert second.switched is True

    decode = controller.select(np.array([1, 1], dtype=np.int32))
    assert decode.num_ubatches == 1

    prefill_again = controller.select(prefill)
    assert prefill_again.previous_m == best_m
    assert prefill_again.num_ubatches == best_m
    assert prefill_again.switched is False


def test_non_safe_candidate_must_beat_safe_m_by_full_margin():
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_switch_threshold_pct=99.0,
        ),
        model_config=_model_config(7),
    )
    prefill = np.array([1024, 1024], dtype=np.int32)

    decision = controller.select(prefill)

    assert min(
        decision.candidate_scores,
        key=lambda score: score["robust_ms"],
    )["m"] > 1
    assert decision.num_ubatches == 1
    assert decision.reason == "safe_m_gain_guard"


def test_measured_regret_falls_back_to_safe_m_and_cools_candidate(
    tmp_path,
):
    trace_path = tmp_path / "adaptive.jsonl"
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_max_size=2,
            adaptive_ubatch_switch_threshold_pct=0.0,
            adaptive_ubatch_trace_path=str(trace_path),
        ),
        model_config=_model_config(7),
    )
    prefill = np.array([1024, 1024], dtype=np.int32)

    safe = controller.select(prefill)
    assert safe.num_ubatches == 1
    controller.observe(safe, forward_ms=_candidate(safe, 1)["prior_ms"])

    candidate = controller.select(prefill)
    assert candidate.num_ubatches == 2
    controller.observe(
        candidate,
        forward_ms=_candidate(candidate, 2)["prior_ms"] * 4.0,
    )

    fallback = controller.select(prefill)
    assert fallback.num_ubatches == 1
    assert _candidate(fallback, 2)["rejected"] is True
    assert _candidate(fallback, 2)["rejection_reason"] == "cooldown"

    controller.close_trace()
    observations = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "adaptive_ubatch_observation"
    ]
    assert observations[-1]["bad_vs_safe"] is True
    assert observations[-1]["relative_safe_regret_pct"] > 0


def test_non_safe_queue_stall_falls_back_without_scenario_hardcoding(
    tmp_path,
):
    trace_path = tmp_path / "adaptive.jsonl"
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_enable_queue_guard=True,
            adaptive_ubatch_max_size=2,
            adaptive_ubatch_trace_path=str(trace_path),
        ),
        model_config=_model_config(7),
    )
    bucket = (
        "7b",
        "prefill",
        "large",
    )
    scores = (
        {"m": 1, "prior_ms": 100.0, "robust_ms": 100.0},
        {"m": 2, "prior_ms": 100.0, "robust_ms": 90.0},
    )

    def decision(m):
        return AdaptiveUBatchDecision(
            num_ubatches=m,
            predicted_gain_pct=10.0 if m == 2 else 0.0,
            reason="keep_current_best",
            bucket_key=bucket,
            total_tokens=2048,
            online=True,
            num_reqs=16,
            predicted_cost_ms=100.0,
            robust_cost_ms=90.0 if m == 2 else 100.0,
            previous_m=m,
            candidate_scores=scores,
            waiting_count=10,
            running_count=20,
            oldest_wait_ms=1000.0,
            pending_first_token_count=10,
            oldest_first_token_wait_ms=1000.0,
            pending_prefill_tokens=10000,
        )

    # Establish a safe-M pressure reference that drains queued work.
    for _ in range(2):
        controller.observe(
            decision(1),
            forward_ms=100.0,
            next_waiting_count=5,
            next_running_count=18,
            next_oldest_wait_ms=500.0,
            next_pending_first_token_count=5,
            next_oldest_first_token_wait_ms=500.0,
            next_pending_prefill_tokens=5000,
        )

    # M=2 has the same step time but twice leaves the oldest request queued.
    for _ in range(2):
        controller.observe(
            decision(2),
            forward_ms=100.0,
            next_waiting_count=10,
            next_running_count=21,
            next_oldest_wait_ms=1100.0,
            next_pending_first_token_count=10,
            next_oldest_first_token_wait_ms=1100.0,
            next_pending_prefill_tokens=10000,
        )

    fallback = controller.select(
        np.array([1024, 1024], dtype=np.int32),
        waiting_count=10,
        running_count=20,
        oldest_wait_ms=1100.0,
        pending_first_token_count=10,
        oldest_first_token_wait_ms=1100.0,
        pending_prefill_tokens=10000,
    )
    assert fallback.num_ubatches == 1
    assert _candidate(fallback, 2)["rejected"] is True
    assert _candidate(fallback, 2)["rejection_reason"] == "cooldown"

    controller.close_trace()
    observations = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "adaptive_ubatch_observation"
    ]
    assert observations[-1]["bad_queue_progress"] is True
    assert observations[-1]["queue_progress_regret"] > 0.20


def test_calibration_does_not_leak_between_workload_buckets():
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(),
        model_config=_model_config(7),
    )
    decode = controller.select(np.array([1, 1], dtype=np.int32))
    decode_prior = _candidate(decode, 1)["prior_ms"]
    controller.observe(decode, forward_ms=decode_prior * 6.0)

    prefill = controller.select(np.array([1024, 1024], dtype=np.int32))

    assert _candidate(prefill, 1)["calibration_scale"] == pytest.approx(1.0)


def test_candidate_calibration_balances_coverage_and_ignores_cold_samples(
    tmp_path,
):
    trace_path = tmp_path / "adaptive.jsonl"
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_candidate_calibration_observations=2,
            adaptive_ubatch_trace_path=str(trace_path),
        ),
        model_config=_model_config(7),
    )
    prefill = np.array([1024, 1024], dtype=np.int32)

    decisions = []
    for _ in range(11):
        decision = controller.select(prefill)
        decisions.append(decision)
        prior_ms = _candidate(decision, decision.num_ubatches)["prior_ms"]
        controller.observe(decision, forward_ms=prior_ms)

    # Each M gets one ignored compilation-prone sample followed by two
    # accepted calibration samples before score-based selection starts.
    calibration = [d for d in decisions if d.reason == "candidate_calibration"]
    assert [d.num_ubatches for d in calibration] == [
        1, 1, 1,
        2, 2, 2,
        4, 4, 4,
    ]
    assert [
        d.num_ubatches
        for d in decisions
        if d.reason == "paired_safe_refresh"
    ] == [1, 1]
    final = controller.select(prefill)
    assert _candidate(final, 1)["count"] >= 2
    assert _candidate(final, 2)["count"] == 2
    assert _candidate(final, 4)["count"] == 2

    controller.close_trace()
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    ignored = [
        record
        for record in records
        if record["type"] == "adaptive_ubatch_cold_sample_ignored"
    ]
    assert [record["selected_m"] for record in ignored] == [1, 2, 4]


def test_prefill_shapes_share_one_bounded_calibration_budget():
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_candidate_calibration_observations=1,
            adaptive_ubatch_switch_confirmations=1,
        ),
        model_config=_model_config(14),
    )
    workloads = (
        np.array([1024, 1024], dtype=np.int32),
        np.array([2047, 1], dtype=np.int32),
        np.array([1024, 992, *([1] * 32)], dtype=np.int32),
    )

    decisions = []
    for step in range(16):
        decision = controller.select(workloads[step % len(workloads)])
        decisions.append(decision)
        prior_ms = _candidate(
            decision,
            decision.num_ubatches,
        )["prior_ms"]
        controller.observe(decision, forward_ms=prior_ms)

    calibration = [
        decision
        for decision in decisions
        if decision.reason == "candidate_calibration"
    ]
    # One ignored cold sample and one accepted sample per M, shared by all
    # equivalent prefill shapes instead of repeated per query/composition.
    assert [decision.num_ubatches for decision in calibration] == [
        1,
        1,
        2,
        2,
        4,
        4,
    ]
    assert len({decision.bucket_key for decision in decisions}) == 1


def test_measured_prefill_gain_is_retained_while_decode_stays_safe():
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_candidate_calibration_observations=1,
            adaptive_ubatch_switch_confirmations=1,
            adaptive_ubatch_switch_threshold_pct=5.0,
        ),
        model_config=_model_config(14),
    )
    prefill = np.array([1024, 1024], dtype=np.int32)

    for _ in range(12):
        decision = controller.select(prefill)
        prior_ms = _candidate(
            decision,
            decision.num_ubatches,
        )["prior_ms"]
        speed_factor = {
            1: 1.0,
            2: 0.8,
            4: 0.5,
        }[decision.num_ubatches]
        controller.observe(
            decision,
            forward_ms=prior_ms * speed_factor,
        )

    retained = controller.select(prefill)
    decode = controller.select(np.array([1, 1], dtype=np.int32))

    assert retained.num_ubatches == 4
    assert decode.num_ubatches == 1


def test_dynamic_mode_periodically_refreshes_safe_reference():
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_candidate_calibration_observations=1,
            adaptive_ubatch_enable_exploration=True,
            adaptive_ubatch_exploration_interval_steps=2,
            adaptive_ubatch_exploration_stable_steps=1,
            adaptive_ubatch_max_size=2,
            adaptive_ubatch_switch_confirmations=1,
        ),
        model_config=_model_config(14),
    )
    prefill = np.array([1024, 1024], dtype=np.int32)

    decisions = []
    for _ in range(12):
        decision = controller.select(prefill)
        decisions.append(decision)
        prior_ms = _candidate(
            decision,
            decision.num_ubatches,
        )["prior_ms"]
        controller.observe(
            decision,
            forward_ms=prior_ms * (
                0.5 if decision.num_ubatches == 2 else 1.0
            ),
        )

    assert any(
        decision.num_ubatches == 2
        for decision in decisions
    )
    assert any(
        decision.reason == "periodic_safe_refresh"
        for decision in decisions
    )

    # Simulate a QPS/load change that makes the previously profitable M=2
    # execution regress. The first measured bad execution must return the next
    # decision to the safe candidate.
    for _ in range(20):
        drifted = controller.select(prefill)
        prior_ms = _candidate(
            drifted,
            drifted.num_ubatches,
        )["prior_ms"]
        controller.observe(
            drifted,
            forward_ms=prior_ms * (
                2.0 if drifted.num_ubatches == 2 else 1.0
            ),
        )
        if drifted.num_ubatches == 2:
            break
    else:
        pytest.fail("dynamic revalidation never sampled M=2")

    fallback = controller.select(prefill)
    assert fallback.num_ubatches == 1
    assert _candidate(fallback, 2)["rejected"] is True


def test_runtime_failure_event_is_distinct_and_safe_m_is_not_cooled_down(
    tmp_path,
):
    trace_path = tmp_path / "adaptive.jsonl"
    controller = AdaptiveUBatchController(
        parallel_config=_parallel_config(
            adaptive_ubatch_trace_path=str(trace_path),
        ),
        model_config=_model_config(7),
    )
    decision = controller.select(np.array([1024, 1024], dtype=np.int32))

    controller.observe_failure(decision, reason="test_failure")
    next_decision = controller.select(np.array([1024, 1024], dtype=np.int32))

    controller.close_trace()
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    failure = next(
        record
        for record in records
        if record["type"] == "adaptive_ubatch_runtime_failure"
    )
    assert failure["reason"] == "test_failure"
    assert failure["affects_cooldown"] is False
    assert _candidate(next_decision, 1)["rejected"] is False
