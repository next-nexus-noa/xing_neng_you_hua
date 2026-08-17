# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.adaptive_ubatch import (
    AdaptiveUBatchController,
    AdaptiveUBatchDecision,
    CandidateScore,
    ContextualValidationState,
    PendingContextualOutcome,
    WorkloadBucket,
    extract_adaptive_ubatch_features,
)


def _parallel_config(**overrides):
    values = {
        "adaptive_ubatch_bad_threshold_pct": 10.0,
        "adaptive_ubatch_candidate_calibration_observations": 0,
        "adaptive_ubatch_cold_start_penalty_ratio": 0.0,
        "adaptive_ubatch_cooldown_steps": 8,
        "adaptive_ubatch_enable_exploration": False,
        "adaptive_ubatch_ewma_alpha": 1.0,
        "adaptive_ubatch_failure_cooldown_steps": 8,
        "adaptive_ubatch_max_calibration_scale": 8.0,
        "adaptive_ubatch_max_size": 4,
        "adaptive_ubatch_max_uncertainty_ratio": 0.5,
        "adaptive_ubatch_max_exploration_regret_pct": 5.0,
        "adaptive_ubatch_queue_growth_threshold": 2,
        "adaptive_ubatch_queue_safety_enabled": True,
        "adaptive_ubatch_regret_budget_pct": 2.0,
        "adaptive_ubatch_regret_window_steps": 64,
        "adaptive_ubatch_context_min_observations": 3,
        "adaptive_ubatch_context_forgetting_factor": 0.98,
        "adaptive_ubatch_context_change_threshold": 0.35,
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


def _validation_boundary_decision(state):
    return AdaptiveUBatchDecision(
        num_ubatches=(
            state.candidate_m if state.phase == "candidate" else 1
        ),
        predicted_gain_pct=0.0,
        reason=f"test_{state.phase}",
        bucket_key=state.target_bucket,
        total_tokens=2048,
        validation_window_id=state.window_id,
        validation_phase=state.phase,
        validation_boundary=True,
        validation_target_bucket=state.target_bucket,
        validation_target_step=True,
        validation_stage=state.stage,
        validation_experiment_id=state.experiment_id,
    )


@pytest.mark.parametrize(
    ("model_billions", "expected_model_bucket"),
    ((3, "medium"), (7, "medium"), (14, "large")),
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


def test_candidate_calibration_balances_coverage_and_marks_transition_samples(
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

    # Each M gets one transition-only sample followed by two accepted
    # steady-state samples before score-based selection starts.
    calibration = [d for d in decisions if d.reason == "candidate_calibration"]
    assert [d.num_ubatches for d in calibration] == [
        1, 1, 1,
        2, 2, 2,
        4, 4, 4,
    ]
    final = controller.select(prefill)
    assert _candidate(final, 1)["count"] >= 2
    assert _candidate(final, 2)["count"] >= 2
    assert _candidate(final, 4)["count"] >= 2

    controller.close_trace()
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    transitions = [
        record
        for record in records
        if record["type"] == "adaptive_ubatch_transition_sample"
    ]
    assert [record["selected_m"] for record in transitions] == [1, 2, 4]


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


def _contextual_controller(**overrides):
    values = {
        "adaptive_ubatch_candidate_calibration_observations": 0,
        "adaptive_ubatch_cold_start_penalty_ratio": 0.0,
        "adaptive_ubatch_context_min_observations": 1,
        "adaptive_ubatch_exploration_interval_steps": 2,
        "adaptive_ubatch_exploration_stable_steps": 1,
        "adaptive_ubatch_max_size": 2,
        "adaptive_ubatch_mode": "contextual_safe",
        "adaptive_ubatch_risk_kappa": 0.0,
        "adaptive_ubatch_switch_confirmations": 1,
    }
    values.update(overrides)
    return AdaptiveUBatchController(
        parallel_config=_parallel_config(**values),
        model_config=_model_config(7),
    )


def _commit_validation_phase(
    controller,
    state,
    *,
    scheduler_steps,
    target_steps,
    output_tokens,
):
    controller._record_validation_context(
        state,
        (1.0,) + (0.0,) * 11,
    )
    controller.observe_service_window(
        _validation_boundary_decision(state),
        elapsed_ms=1000.0,
        output_tokens=output_tokens,
        completed_reqs=0,
        scheduler_steps=scheduler_steps,
        target_steps=target_steps,
    )


def test_contextual_validation_rejects_background_mismatched_label():
    controller = _contextual_controller()
    state = ContextualValidationState(
        generation=0,
        target_bucket=("medium", "prefill", "large"),
        candidate_m=2,
        phase="safe_before",
        window_id=1,
        experiment_id=1,
    )
    controller._active_validation = state

    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=8,
        target_steps=8,
        output_tokens=100,
    )
    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=40,
        target_steps=4,
        output_tokens=120,
    )
    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=8,
        target_steps=8,
        output_tokens=100,
    )

    assert controller._contextual_state(2).count == 0
    assert controller._active_validation.phase == "idle"


def test_contextual_validation_requires_repeated_matched_labels():
    controller = _contextual_controller()
    state = ContextualValidationState(
        generation=0,
        target_bucket=("medium", "prefill", "large"),
        candidate_m=2,
        phase="safe_before",
        window_id=1,
        experiment_id=1,
    )
    controller._active_validation = state

    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=8,
        target_steps=8,
        output_tokens=100,
    )
    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=4,
        target_steps=4,
        output_tokens=120,
    )
    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=8,
        target_steps=8,
        output_tokens=100,
    )

    assert controller._contextual_state(2).count == 0
    assert controller._active_validation.phase == "candidate"
    assert controller._active_validation.stage == 0

    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=4,
        target_steps=4,
        output_tokens=120,
    )
    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=8,
        target_steps=8,
        output_tokens=100,
    )

    assert controller._contextual_state(2).count == 1
    assert controller._active_validation.phase == "candidate"
    assert controller._active_validation.stage == 1


def test_contextual_validation_stops_on_severe_first_loss():
    controller = _contextual_controller()
    state = ContextualValidationState(
        generation=0,
        target_bucket=("medium", "prefill", "large"),
        candidate_m=2,
        phase="safe_before",
        window_id=1,
        experiment_id=1,
    )
    controller._active_validation = state

    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=8,
        target_steps=8,
        output_tokens=100,
    )
    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=8,
        target_steps=8,
        output_tokens=80,
    )
    _commit_validation_phase(
        controller,
        state,
        scheduler_steps=8,
        target_steps=8,
        output_tokens=100,
    )

    assert controller._contextual_state(2).count == 0
    assert controller._active_validation.phase == "idle"


def test_contextual_candidate_ignores_unpaired_full_worker_feedback():
    controller = _contextual_controller(
        adaptive_ubatch_online_residual_limit_pct=8.0,
    )
    state = controller._contextual_state(2)
    offline_coefficients = state.offline_coefficients.copy()
    context = (1.0,) + (0.0,) * 11
    controller._pending_contextual_outcome = PendingContextualOutcome(
        bucket_key=("medium", "prefill", "large"),
        selected_m=2,
        context=context,
        prior_ms=100.0,
        actual_ms=10.0,
        baseline_ms=100.0,
        queue_depth=1,
        waiting_reqs=1,
    )

    controller._finalize_contextual_outcome(
        queue_depth=1,
        waiting_reqs=1,
    )

    assert state.count == 0
    np.testing.assert_array_equal(
        state.offline_coefficients,
        offline_coefficients,
    )


def test_contextual_paired_residual_is_bounded_around_offline_prior():
    controller = _contextual_controller(
        adaptive_ubatch_online_residual_limit_pct=8.0,
    )
    state = controller._contextual_state(2)
    context = np.asarray((1.0,) + (0.0,) * 11)

    state.update(
        context=context,
        target=10.0,
        forgetting_factor=1.0,
        alpha=1.0,
        step_id=1,
    )
    _, online_residual, _ = state.predict_components(context)

    assert abs(online_residual) <= np.log1p(0.08)


def _observe_contextual_step(controller, workload, *, candidate_scale):
    decision = controller.select(workload, queue_depth=4, waiting_reqs=4)
    selected = _candidate(decision, decision.num_ubatches)
    scale = candidate_scale if decision.num_ubatches > 1 else 1.0
    controller.observe(decision, forward_ms=selected["prior_ms"] * scale)
    return decision


def _apply_contextual_after_safe_anchor(
    controller,
    *,
    proposed,
    safe_score,
    candidate_scores,
    bucket,
    context,
):
    selected, reason, *_ = controller._apply_contextual_safety(
        proposed=proposed,
        safe_score=safe_score,
        candidate_scores=candidate_scores,
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )
    assert selected.m == safe_score.m
    assert reason == "contextual_probe_pre_anchor"
    regime = controller._regime_state(bucket)
    regime.safe_anchor_scale = 1.0
    regime.safe_anchor_context = tuple(context)
    regime.safe_anchor_step = controller._step_id + 1
    regime.safe_anchor_generation = regime.generation
    controller._step_id += 1
    return controller._apply_contextual_safety(
        proposed=proposed,
        safe_score=safe_score,
        candidate_scores=candidate_scores,
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )


def test_contextual_safe_learns_and_preserves_a_beneficial_candidate():
    controller = _contextual_controller()
    prefill = np.array([1024, 1024], dtype=np.int32)

    decisions = [
        _observe_contextual_step(controller, prefill, candidate_scale=0.7)
        for _ in range(20)
    ]
    final = controller.select(prefill, queue_depth=4, waiting_reqs=4)

    assert "contextual_exploration" in {d.reason for d in decisions}
    assert controller._contextual_state(1).count >= 1
    assert controller._contextual_state(2).count >= 1
    assert 2 in {decision.num_ubatches for decision in decisions}
    assert final.reason in {
        "contextual_exposure_guard",
        "contextual_exposure_validation",
        "contextual_bounded_gain",
    }
    assert final.contextual_gain_lcb_pct > 0
    assert _candidate(final, 2)["contextual_relative_ratio"] < 1.0


def test_contextual_exploration_uses_stable_visits_across_other_buckets():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=3,
        adaptive_ubatch_exploration_interval_steps=1,
        adaptive_ubatch_exploration_stable_steps=3,
    )
    prefill = np.array([1024, 1024], dtype=np.int32)
    decode = np.array([1, 1], dtype=np.int32)

    prefill_decisions = []
    for _ in range(8):
        prefill_decisions.append(
            _observe_contextual_step(
                controller,
                prefill,
                candidate_scale=0.7,
            )
        )
        _observe_contextual_step(
            controller,
            decode,
            candidate_scale=1.0,
        )

    prefill_bucket = WorkloadBucket.from_key(
        ("medium", "prefill", "large")
    )
    assert controller._regime_state(
        prefill_bucket
    ).stable_observations >= 3
    assert controller._stable_bucket_steps == 1
    assert "contextual_exploration" in {
        decision.reason for decision in prefill_decisions
    }


def test_contextual_regime_change_resets_only_changed_bucket_stability():
    controller = _contextual_controller(
        adaptive_ubatch_context_change_threshold=0.01,
        adaptive_ubatch_exploration_stable_steps=3,
    )
    prefill = np.array([1024, 1024], dtype=np.int32)
    decode = np.array([1, 1], dtype=np.int32)
    changed = np.array([1800, 200], dtype=np.int32)

    for _ in range(4):
        controller.select(prefill, queue_depth=4, waiting_reqs=4)
        controller.select(decode, queue_depth=4, waiting_reqs=4)

    prefill_bucket = WorkloadBucket.from_key(
        ("medium", "prefill", "large")
    )
    decode_bucket = WorkloadBucket.from_key(("medium", "decode", "small"))
    decode_stability = controller._regime_state(
        decode_bucket
    ).stable_observations

    decisions = [
        controller.select(changed, queue_depth=12, waiting_reqs=10)
        for _ in range(4)
    ]

    assert "contextual_regime_warmup" in {
        decision.reason for decision in decisions
    }
    assert controller._regime_state(prefill_bucket).generation >= 1
    assert (
        controller._regime_state(decode_bucket).stable_observations
        == decode_stability
    )


def test_contextual_safe_can_choose_m2_when_base_policy_proposes_m4():
    controller = _contextual_controller(adaptive_ubatch_risk_kappa=1.0)
    context = np.array((1.0,) + (0.0,) * 10, dtype=np.float64)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    m2 = CandidateScore(2, 80.0, 0.0, 80.0, 0.0, 80.0, 3)
    m4 = CandidateScore(4, 70.0, 0.0, 70.0, 0.0, 70.0, 3)
    for _ in range(3):
        controller._contextual_state(1).update(
            context=np.append(context, 0.0),
            target=0.0,
            forgetting_factor=0.98,
            alpha=0.2,
            step_id=controller._step_id,
        )
        controller._contextual_state(2).update(
            context=np.append(context, np.log(0.8)),
            target=np.log(0.8),
            forgetting_factor=0.98,
            alpha=0.2,
            step_id=controller._step_id,
        )
        controller._contextual_state(4).update(
            context=np.append(context, np.log(0.7)),
            target=np.log(1.1),
            forgetting_factor=0.98,
            alpha=0.2,
            step_id=controller._step_id,
        )

    bucket = WorkloadBucket("medium", "prefill", "large")
    exposure = controller._exposure_state(bucket, 2)
    exposure.stage = len(controller._exposure_ratios())
    exposure.credit = 1.0
    exposure.validated = True
    selected, reason, gain, _, _, evaluations = (
        controller._apply_contextual_safety(
            proposed=m4,
            safe_score=safe,
            candidate_scores=[safe, m2, m4],
            bucket=bucket,
            context=tuple(context),
            regime_warmup=False,
        )
    )

    assert selected.m == 2
    assert reason == "contextual_bounded_gain"
    assert gain > 5.0
    assert evaluations[2]["contextual_relative_ratio"] < 1.0
    assert evaluations[4]["contextual_relative_ratio"] > 1.0

    controller._regret_state(bucket).add(
        regret_ms=10.0,
        baseline_ms=100.0,
        window_steps=64,
    )
    protected, protected_reason, *_ = controller._apply_contextual_safety(
        proposed=m4,
        safe_score=safe,
        candidate_scores=[safe, m2, m4],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )
    assert protected.m == 1
    assert protected_reason == "contextual_regret_budget"


def test_contextual_safe_blocks_after_exceeding_regret_budget():
    controller = _contextual_controller(
        adaptive_ubatch_regret_budget_pct=2.0,
        adaptive_ubatch_regret_window_steps=8,
    )
    prefill = np.array([1024, 1024], dtype=np.int32)
    bucket = WorkloadBucket.from_key(("medium", "prefill", "large"))
    controller._regret_state(bucket).add(
        regret_ms=1000.0,
        baseline_ms=100.0,
        window_steps=8,
    )
    protected = controller.select(prefill, queue_depth=4, waiting_reqs=4)

    assert protected.num_ubatches == 1
    assert protected.reason == "contextual_regret_budget"
    assert protected.contextual_regret_pct >= 2.0


def test_contextual_candidate_learns_direct_ratio_to_safe_baseline():
    controller = _contextual_controller()
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = (1.0,) + (0.0,) * 10
    controller._pending_contextual_outcome = PendingContextualOutcome(
        bucket_key=bucket.as_tuple(),
        selected_m=2,
        context=context,
        prior_ms=50.0,
        actual_ms=80.0,
        baseline_ms=100.0,
        queue_depth=4,
        waiting_reqs=4,
    )

    controller._finalize_contextual_outcome(
        queue_depth=4,
        waiting_reqs=4,
    )
    predicted_log_ratio, _ = controller._contextual_state(2).predict(
        np.asarray(context)
    )

    assert predicted_log_ratio < 0.0
    assert np.exp(predicted_log_ratio) < 1.0


def test_contextual_outlier_is_clipped_and_only_temporarily_cools_arm():
    controller = _contextual_controller(
        adaptive_ubatch_failure_cooldown_steps=8,
        adaptive_ubatch_max_correction_ratio=0.3,
        adaptive_ubatch_max_exploration_regret_pct=5.0,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = (1.0,) + (0.0,) * 10
    controller._pending_contextual_outcome = PendingContextualOutcome(
        bucket_key=bucket.as_tuple(),
        selected_m=2,
        context=context,
        prior_ms=50.0,
        actual_ms=300.0,
        baseline_ms=100.0,
        queue_depth=4,
        waiting_reqs=4,
    )

    controller._finalize_contextual_outcome(
        queue_depth=4,
        waiting_reqs=4,
    )

    predicted_log_ratio, _ = controller._contextual_state(2).predict(
        np.asarray(context)
    )
    assert predicted_log_ratio <= np.log(1.3)
    assert controller._regret_state(bucket).regret_pct() == pytest.approx(200.0)
    assert controller._cooldown_until[(bucket.as_tuple(), 2)] == 8
    assert (bucket.as_tuple(), 4) not in controller._cooldown_until

    for _ in range(2):
        controller._pending_contextual_outcome = PendingContextualOutcome(
            bucket_key=bucket.as_tuple(),
            selected_m=1,
            context=context,
            prior_ms=100.0,
            actual_ms=100.0,
            baseline_ms=100.0,
            queue_depth=4,
            waiting_reqs=4,
        )
        controller._finalize_contextual_outcome(
            queue_depth=4,
            waiting_reqs=4,
        )

    assert controller._regret_state(bucket).regret_pct() > 2.0


def test_contextual_cold_exploration_stages_m2_before_m4():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=1,
        adaptive_ubatch_exploration_interval_steps=1,
        adaptive_ubatch_exploration_stable_steps=1,
        adaptive_ubatch_max_size=4,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    m2 = CandidateScore(2, 99.0, 0.0, 99.0, 0.0, 99.0, 0)
    m4 = CandidateScore(4, 70.0, 0.0, 70.0, 0.0, 70.0, 0)
    controller._contextual_state(1).update(
        context=context,
        target=0.0,
        forgetting_factor=0.98,
        alpha=0.2,
        step_id=0,
    )
    controller._regime_state(bucket).stable_observations = 1
    controller._step_id = 1

    selected, reason, *_ = _apply_contextual_after_safe_anchor(
        controller,
        proposed=m4,
        safe_score=safe,
        candidate_scores=[safe, m2, m4],
        bucket=bucket,
        context=context,
    )

    assert selected.m == 2
    assert reason == "contextual_exploration"


def test_contextual_proven_m2_does_not_starve_cold_m4():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=3,
        adaptive_ubatch_exploration_interval_steps=1,
        adaptive_ubatch_exploration_stable_steps=1,
        adaptive_ubatch_max_size=4,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    m2 = CandidateScore(2, 80.0, 0.0, 80.0, 0.0, 80.0, 3)
    m4 = CandidateScore(4, 70.0, 0.0, 70.0, 0.0, 70.0, 0)
    for _ in range(3):
        controller._contextual_state(1).update(
            context=np.append(context, 0.0),
            target=0.0,
            forgetting_factor=0.98,
            alpha=0.2,
            step_id=0,
        )
        controller._contextual_state(2).update(
            context=np.append(context, np.log(0.8)),
            target=np.log(0.8),
            forgetting_factor=0.98,
            alpha=0.2,
            step_id=0,
        )
    controller._regime_state(bucket).stable_observations = 1
    controller._step_id = 1

    selected, reason, *_ = _apply_contextual_after_safe_anchor(
        controller,
        proposed=m2,
        safe_score=safe,
        candidate_scores=[safe, m2, m4],
        bucket=bucket,
        context=context,
    )

    assert selected.m == 4
    assert reason == "contextual_exploration"


def test_contextual_safe_detects_a_dynamic_regime_change():
    controller = _contextual_controller(
        adaptive_ubatch_context_change_threshold=0.01,
        adaptive_ubatch_exploration_stable_steps=3,
    )
    steady = np.array([1024, 1024], dtype=np.int32)
    changed = np.array([1800, 200], dtype=np.int32)

    for _ in range(10):
        _observe_contextual_step(controller, steady, candidate_scale=0.7)
    decisions = [
        controller.select(changed, queue_depth=12, waiting_reqs=10)
        for _ in range(4)
    ]

    assert all(decision.num_ubatches == 1 for decision in decisions)
    assert "contextual_regime_warmup" in {
        decision.reason for decision in decisions
    }


def test_contextual_safe_requires_repeated_queue_regression():
    controller = _contextual_controller(
        adaptive_ubatch_failure_cooldown_steps=8,
        adaptive_ubatch_queue_growth_threshold=2,
        adaptive_ubatch_switch_confirmations=2,
    )
    bucket = WorkloadBucket.from_key(("medium", "prefill", "large"))
    context = (1.0,) + (0.0,) * 10

    for index in range(2):
        controller._pending_contextual_outcome = PendingContextualOutcome(
            bucket_key=bucket.as_tuple(),
            selected_m=2,
            context=context,
            prior_ms=100.0,
            actual_ms=90.0,
            baseline_ms=100.0,
            queue_depth=4 + index * 4,
            waiting_reqs=4 + index * 4,
        )
        controller._step_id += 1
        controller._finalize_contextual_outcome(
            queue_depth=8 + index * 4,
            waiting_reqs=8 + index * 4,
        )

    assert controller._cooldown_until[(bucket.as_tuple(), 2)] > (
        controller._step_id
    )


def test_contextual_decision_payload_round_trip():
    decision = AdaptiveUBatchDecision(
        num_ubatches=2,
        predicted_gain_pct=7.0,
        reason="contextual_proven_gain",
        bucket_key=("medium", "prefill", "large"),
        total_tokens=2048,
        context_vector=(1.0, 0.5),
        contextual_baseline_ms=12.5,
        contextual_gain_lcb_pct=4.0,
        contextual_regret_pct=1.25,
    )

    restored = AdaptiveUBatchDecision.from_payload(decision.to_payload())

    assert restored.context_vector == decision.context_vector
    assert restored.contextual_baseline_ms == pytest.approx(12.5)
    assert restored.contextual_gain_lcb_pct == pytest.approx(4.0)
    assert restored.contextual_regret_pct == pytest.approx(1.25)


def test_contextual_decision_reports_safe_arm_observation_count():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=3,
    )
    tokens = np.array([1024, 1024], dtype=np.int32)

    decision = controller.select(
        tokens,
        queue_depth=8,
        waiting_reqs=8,
    )

    safe = _candidate(decision, 1)
    assert safe["contextual_role"] == "safe"
    assert safe["contextual_count"] == 0


def _seed_contextual_response(controller, m, context, log_ratio, count=3):
    model_context = np.append(context, log_ratio if m > 1 else 0.0)
    for _ in range(count):
        controller._contextual_state(m).update(
            context=model_context,
            target=log_ratio,
            forgetting_factor=0.98,
            alpha=0.2,
            step_id=0,
        )


def test_contextual_queue_trend_blocks_marginal_but_keeps_strong_gain():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=3,
        adaptive_ubatch_min_gain_pct=5.0,
        adaptive_ubatch_risk_kappa=0.0,
        adaptive_ubatch_switch_confirmations=2,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    marginal = CandidateScore(2, 90.0, 0.0, 90.0, 0.0, 90.0, 3)
    strong = CandidateScore(4, 70.0, 0.0, 70.0, 0.0, 70.0, 3)
    _seed_contextual_response(controller, 1, context, 0.0)
    _seed_contextual_response(controller, 2, context, np.log(0.90))
    _seed_contextual_response(controller, 4, context, np.log(0.70))
    exposure = controller._exposure_state(bucket, 4)
    exposure.stage = len(controller._exposure_ratios())
    exposure.credit = 1.0
    exposure.validated = True
    for _ in range(3):
        controller._regret_state(bucket).add_safe_queue_growth(
            growth=1.0,
            window_steps=3,
        )

    selected, reason, *_, evaluations = controller._apply_contextual_safety(
        proposed=marginal,
        safe_score=safe,
        candidate_scores=[safe, marginal],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )
    assert selected.m == 1
    assert reason == "contextual_queue_gain_guard"
    assert evaluations[2]["contextual_required_gain_pct"] == pytest.approx(
        5.0 + 5.0 * np.sqrt(2.0)
    )

    selected, reason, *_ = controller._apply_contextual_safety(
        proposed=strong,
        safe_score=safe,
        candidate_scores=[safe, marginal, strong],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )
    assert selected.m == 4
    assert reason == "contextual_bounded_gain"


def test_contextual_stable_regime_does_not_repeat_uncertain_exploration():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=3,
        adaptive_ubatch_exploration_interval_steps=1,
        adaptive_ubatch_exploration_stable_steps=1,
        adaptive_ubatch_min_gain_pct=5.0,
        adaptive_ubatch_risk_kappa=1.0,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    uncertain = CandidateScore(2, 95.0, 0.0, 95.0, 0.0, 95.0, 3)
    _seed_contextual_response(controller, 1, context, 0.0)
    state = controller._contextual_state(2)
    state.count = 3
    state.coefficients.fill(0.0)
    state.coefficients[0] = np.log(0.90)
    state.covariance = np.eye(state.dimension) * 0.1
    state.residual_variance = 0.05
    controller._regime_state(bucket).stable_observations = 1
    controller._step_id = 100

    selected, reason, *_ = controller._apply_contextual_safety(
        proposed=uncertain,
        safe_score=safe,
        candidate_scores=[safe, uncertain],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )

    assert selected.m == 1
    assert reason == "contextual_insufficient_evidence"
    assert controller._regime_state(bucket).exploration_attempts == {}


def test_contextual_underobserved_arm_is_eliminated_when_ucb_is_negative():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=3,
        adaptive_ubatch_exploration_interval_steps=1,
        adaptive_ubatch_exploration_stable_steps=1,
        adaptive_ubatch_min_gain_pct=5.0,
        adaptive_ubatch_risk_kappa=1.0,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    candidate = CandidateScore(2, 70.0, 0.0, 70.0, 0.0, 70.0, 1)
    _seed_contextual_response(controller, 1, context, 0.0, count=3)
    state = controller._contextual_state(2)
    state.count = 1
    state.coefficients.fill(0.0)
    state.coefficients[0] = np.log(1.5)
    state.covariance = np.eye(state.dimension) * 1e-6
    state.residual_variance = 1e-6
    controller._regime_state(bucket).stable_observations = 1
    controller._step_id = 100

    selected, reason, *_, evaluations = controller._apply_contextual_safety(
        proposed=candidate,
        safe_score=safe,
        candidate_scores=[safe, candidate],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )

    assert selected.m == 1
    assert reason == "contextual_insufficient_evidence"
    assert evaluations[2]["contextual_gain_ucb_pct"] < 0.0
    assert controller._regime_state(bucket).probe_m is None


def test_contextual_probe_uses_a_contiguous_lease():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=3,
        adaptive_ubatch_exploration_interval_steps=1,
        adaptive_ubatch_exploration_stable_steps=1,
        adaptive_ubatch_min_gain_pct=0.0,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    candidate = CandidateScore(2, 80.0, 0.0, 80.0, 0.0, 80.0, 0)
    _seed_contextual_response(controller, 1, context, 0.0, count=3)
    controller._regime_state(bucket).stable_observations = 1
    controller._step_id = 100

    first, first_reason, *_ = controller._apply_contextual_safety(
        proposed=candidate,
        safe_score=safe,
        candidate_scores=[safe, candidate],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )
    regime = controller._regime_state(bucket)
    regime.safe_anchor_scale = 1.0
    regime.safe_anchor_context = tuple(context)
    regime.safe_anchor_step = controller._step_id + 1
    regime.safe_anchor_generation = regime.generation
    controller._step_id += 1
    second, second_reason, *_ = controller._apply_contextual_safety(
        proposed=candidate,
        safe_score=safe,
        candidate_scores=[safe, candidate],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )
    controller._step_id += 1
    third, third_reason, *_ = controller._apply_contextual_safety(
        proposed=safe,
        safe_score=safe,
        candidate_scores=[safe, candidate],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )

    assert first.m == 1
    assert first_reason == "contextual_probe_pre_anchor"
    assert second.m == third.m == 2
    assert second_reason == "contextual_exploration"
    assert third_reason == "contextual_probe_lease"
    assert controller._regime_state(bucket).probe_remaining == 2


def test_contextual_transition_uses_measured_budget_not_prediction_error():
    controller = _contextual_controller(
        adaptive_ubatch_bad_threshold_pct=20.0,
        adaptive_ubatch_context_min_observations=1,
        adaptive_ubatch_exploration_interval_steps=1,
        adaptive_ubatch_exploration_stable_steps=1,
        adaptive_ubatch_min_gain_pct=0.0,
    )
    workload = np.array([1024, 1024], dtype=np.int32)
    probe = None
    for _ in range(20):
        decision = controller.select(
            workload,
            queue_depth=4,
            waiting_reqs=4,
        )
        if decision.reason == "contextual_exploration":
            probe = decision
            break
        selected = _candidate(decision, decision.num_ubatches)
        controller.observe(decision, forward_ms=selected["prior_ms"])

    assert probe is not None
    assert probe.num_ubatches > 1
    catastrophic_ms = max(
        float(probe.predicted_cost_ms) * 2.0,
        float(probe.contextual_baseline_ms) * 2.0,
    )
    controller.observe(probe, forward_ms=catastrophic_ms)
    bucket = WorkloadBucket.from_key(probe.bucket_key)
    regime = controller._regime_state(bucket)

    # A large analytical prediction miss alone must not kill an arm before
    # its transition cost is compared with the measured M=1 safety budget.
    assert regime.probe_m == probe.num_ubatches
    assert controller._decision_state(bucket).current_m == probe.num_ubatches
    assert (bucket.as_tuple(), probe.num_ubatches) not in (
        controller._cooldown_until
    )

    protected = controller.select(
        workload,
        queue_depth=4,
        waiting_reqs=4,
    )
    assert protected.num_ubatches == 1
    assert protected.reason == "contextual_regret_budget"
    assert protected.contextual_regret_pct > 2.0
    assert controller._regime_state(bucket).probe_m is None


def test_contextual_dynamic_regime_allows_one_bounded_revalidation():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=3,
        adaptive_ubatch_exploration_interval_steps=1,
        adaptive_ubatch_exploration_stable_steps=1,
        adaptive_ubatch_risk_kappa=1.0,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    uncertain = CandidateScore(2, 95.0, 0.0, 95.0, 0.0, 95.0, 3)
    _seed_contextual_response(controller, 1, context, 0.0)
    state = controller._contextual_state(2)
    state.count = 3
    state.coefficients.fill(0.0)
    state.coefficients[0] = np.log(0.90)
    state.covariance = np.eye(state.dimension) * 0.1
    state.residual_variance = 0.05
    regime = controller._regime_state(bucket)
    regime.generation = 1
    regime.stable_observations = 1
    controller._step_id = 100

    selected, reason, *_ = _apply_contextual_after_safe_anchor(
        controller,
        proposed=uncertain,
        safe_score=safe,
        candidate_scores=[safe, uncertain],
        bucket=bucket,
        context=context,
    )
    assert selected.m == 2
    assert reason == "contextual_exploration"
    assert regime.exploration_attempts[2] == 1

    lease_reasons = []
    for _ in range(3):
        controller._step_id += 1
        selected, reason, *_ = controller._apply_contextual_safety(
            proposed=uncertain,
            safe_score=safe,
            candidate_scores=[safe, uncertain],
            bucket=bucket,
            context=tuple(context),
            regime_warmup=False,
        )
        lease_reasons.append(reason)
        assert selected.m == 2
    assert lease_reasons == ["contextual_probe_lease"] * 3

    controller._step_id += 1
    selected, reason, *_ = controller._apply_contextual_safety(
        proposed=uncertain,
        safe_score=safe,
        candidate_scores=[safe, uncertain],
        bucket=bucket,
        context=tuple(context),
        regime_warmup=False,
    )
    assert selected.m == 1
    assert reason == "contextual_probe_post_anchor"


def test_contextual_safe_anchor_corrects_the_candidate_baseline():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=1,
        adaptive_ubatch_max_correction_ratio=0.5,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    candidate = CandidateScore(2, 90.0, 0.0, 90.0, 0.0, 90.0, 3)
    _seed_contextual_response(controller, 1, context, 0.0, count=1)
    _seed_contextual_response(
        controller,
        2,
        context,
        np.log(0.9),
        count=1,
    )
    regime = controller._regime_state(bucket)
    regime.safe_anchor_scale = 1.5
    regime.safe_anchor_context = tuple(context)
    regime.safe_anchor_step = controller._step_id
    regime.safe_anchor_generation = regime.generation

    _, _, _, baseline_ms, _, evaluations = (
        controller._apply_contextual_safety(
            proposed=candidate,
            safe_score=safe,
            candidate_scores=[safe, candidate],
            bucket=bucket,
            context=tuple(context),
            regime_warmup=False,
        )
    )

    assert baseline_ms == pytest.approx(150.0)
    assert evaluations[2]["contextual_baseline_source"] == (
        "paired_safe_anchor"
    )


def test_contextual_candidate_regret_is_isolated_by_m():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=1,
        adaptive_ubatch_max_size=4,
        adaptive_ubatch_regret_budget_pct=2.0,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    m2 = CandidateScore(2, 80.0, 0.0, 80.0, 0.0, 80.0, 3)
    m4 = CandidateScore(4, 70.0, 0.0, 70.0, 0.0, 70.0, 3)
    _seed_contextual_response(controller, 1, context, 0.0, count=1)
    _seed_contextual_response(
        controller, 2, context, np.log(0.8), count=1
    )
    _seed_contextual_response(
        controller, 4, context, np.log(0.7), count=1
    )
    exposure = controller._exposure_state(bucket, 4)
    exposure.stage = len(controller._exposure_ratios())
    exposure.credit = 1.0
    exposure.validated = True
    controller._arm_regret_state(bucket, 2).add(
        regret_ms=5.0,
        baseline_ms=100.0,
        window_steps=64,
    )

    selected, reason, *_, evaluations = (
        controller._apply_contextual_safety(
            proposed=m2,
            safe_score=safe,
            candidate_scores=[safe, m2, m4],
            bucket=bucket,
            context=tuple(context),
            regime_warmup=False,
        )
    )

    assert selected.m == 4
    assert reason == "contextual_bounded_gain"
    assert evaluations[2]["contextual_candidate_regret_pct"] >= 2.0
    assert evaluations[4]["contextual_candidate_regret_pct"] == 0.0


def test_contextual_queue_trend_updates_without_timing_feedback():
    controller = _contextual_controller(
        adaptive_ubatch_exploration_stable_steps=2,
    )
    workload = np.array([1024, 1024], dtype=np.int32)

    for queue_depth in range(1, 11):
        controller.select(
            workload,
            queue_depth=queue_depth,
            waiting_reqs=max(0, queue_depth - 2),
        )

    bucket = WorkloadBucket("medium", "prefill", "large")
    state = controller._regret_state(bucket)
    assert len(state.safe_queue_growth) >= 8
    assert state.safe_queue_positive_ratio(min_samples=8) > 0.9


def test_contextual_window_ignores_one_transient_regime_outlier():
    controller = _contextual_controller(
        adaptive_ubatch_context_change_threshold=0.05,
        adaptive_ubatch_exploration_stable_steps=4,
        adaptive_ubatch_switch_confirmations=2,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    steady = (1.0, 0.3, 0.5, 0.4, 0.5, 0.5, 0.95, 0.9, 0.2, 0.0, 0.0)
    outlier = (1.0, 0.3, 0.8, 0.8, 0.8, 0.7, 0.5, 0.5, 0.9, 0.8, 0.8)

    for _ in range(12):
        controller._update_context_regime(
            bucket=bucket,
            context=steady,
        )
    controller._update_context_regime(bucket=bucket, context=outlier)
    for _ in range(8):
        controller._update_context_regime(
            bucket=bucket,
            context=steady,
        )

    state = controller._regime_state(bucket)
    assert state.generation == 0
    assert state.warming_up is False


def test_contextual_candidate_exposure_promotes_and_demotes_by_window():
    controller = _contextual_controller(
        adaptive_ubatch_exposure_stages="0.05,0.10,0.25,0.50",
        adaptive_ubatch_exposure_validation_observations=3,
        adaptive_ubatch_exposure_promotion_gain_pct=2.0,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")

    outcomes = [
        controller._record_exposure_observation(
            bucket=bucket,
            m=2,
            relative_ratio=0.90,
            queue_growth=0.0,
        )
        for _ in range(3)
    ]
    assert outcomes[-1] == (2, "promoted")

    outcomes = [
        controller._record_exposure_observation(
            bucket=bucket,
            m=2,
            relative_ratio=1.05,
            queue_growth=1.0,
        )
        for _ in range(3)
    ]
    assert outcomes[-1] == (1, "demoted")


def test_contextual_proven_candidate_is_rate_limited_at_first_stage():
    controller = _contextual_controller(
        adaptive_ubatch_context_min_observations=1,
        adaptive_ubatch_exposure_stages="0.05,0.10,0.25,0.50",
        adaptive_ubatch_risk_kappa=0.0,
    )
    bucket = WorkloadBucket("medium", "prefill", "large")
    context = np.asarray((1.0,) + (0.0,) * 10)
    safe = CandidateScore(1, 100.0, 0.0, 100.0, 0.0, 100.0, 3)
    candidate = CandidateScore(2, 70.0, 0.0, 70.0, 0.0, 70.0, 3)
    _seed_contextual_response(controller, 1, context, 0.0, count=1)
    _seed_contextual_response(
        controller, 2, context, np.log(0.70), count=1
    )
    exposure = controller._exposure_state(bucket, 2)
    exposure.stage = 1

    selected_ms = []
    reasons = []
    for _ in range(20):
        selected, reason, *_ = controller._apply_contextual_safety(
            proposed=candidate,
            safe_score=safe,
            candidate_scores=[safe, candidate],
            bucket=bucket,
            context=tuple(context),
            regime_warmup=False,
        )
        selected_ms.append(selected.m)
        reasons.append(reason)

    assert selected_ms.count(2) == 1
    assert reasons.count("contextual_exposure_validation") == 1
    assert reasons.count("contextual_exposure_guard") == 19
