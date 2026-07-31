# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.adaptive_ubatch import AdaptiveUBatchController


def _parallel_config(**overrides):
    values = {
        "adaptive_ubatch_bad_threshold_pct": 10.0,
        "adaptive_ubatch_cold_start_penalty_ratio": 0.0,
        "adaptive_ubatch_cooldown_steps": 8,
        "adaptive_ubatch_enable_exploration": False,
        "adaptive_ubatch_ewma_alpha": 1.0,
        "adaptive_ubatch_failure_cooldown_steps": 8,
        "adaptive_ubatch_max_calibration_scale": 8.0,
        "adaptive_ubatch_max_size": 4,
        "adaptive_ubatch_max_uncertainty_ratio": 0.5,
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
