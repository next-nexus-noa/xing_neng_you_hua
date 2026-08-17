# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import numpy as np
import regex as re

CONTEXT_TOKEN_LOG_NORMALIZER = 9.0
CONTEXT_REQUEST_LOG_NORMALIZER = 6.0
CONTEXT_MODEL_LOG_NORMALIZER = 4.0
BASE_CONTEXT_DIMENSION = 11
CONTEXT_DIMENSION = BASE_CONTEXT_DIMENSION + 1

# Hardware-level priors fitted with a robust (Huber) regression over matched
# A/B/A windows collected on 310P, PP=2.  The first eleven terms are the
# workload context below; the final term is the analytical log-cost ratio
# log(T_prior(M) / T_prior(M=1)).  They are only a cold-start prior: live A/B/A
# observations continue to update the RLS state and every candidate still has
# to pass the serving-safety validation before promotion.
_OFFLINE_CONTEXTUAL_PRIORS: dict[int, tuple[tuple[float, ...], float]] = {
    2: (
        (
            -0.299737089232,
            -0.144010915746,
            2.16019768514,
            -0.375071234777,
            -3.09848474303,
            1.01378855296,
            0.297161780647,
            -2.08661637359,
            0.628326920034,
            0.0332283628449,
            -0.151537419964,
            -1.93566372923,
        ),
        0.105,
    ),
    4: (
        (
            -2.37072602373,
            -0.239639175141,
            3.85405548795,
            1.57675101944,
            -2.60437797767,
            1.75511233359,
            0.523231695188,
            -1.5542716623,
            -1.05640850867,
            -0.205775174663,
            0.43471729042,
            -0.068983049764,
        ),
        0.075,
    ),
}

# A policy window is not a counterfactual for M when its share of target
# bucket steps is materially different from the two surrounding M=1 windows.
# This bound removed the +200--675% labels caused by unrelated background work
# while retaining 211 matched observations across all model sizes.
_VALIDATION_TARGET_SHARE_TOLERANCE = 0.10


@dataclass(frozen=True)
class AdaptiveUBatchFeatures:
    total_tokens: int
    num_reqs: int
    max_query_len: int
    decode_tokens: int
    prefill_tokens: int
    prefill_reqs: int
    decode_reqs: int
    prefill_ratio: float
    avg_tokens_per_req: float
    token_imbalance: float
    smallest_request_ratio: float
    model_billions: float
    hidden_size: int
    bucket_key: tuple[str, ...]


@dataclass(frozen=True)
class WorkloadBucket:
    model_bucket: str
    phase_bucket: str
    token_bucket: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (
            self.model_bucket,
            self.phase_bucket,
            self.token_bucket,
        )

    @classmethod
    def from_key(cls, key: tuple[str, ...] | None) -> "WorkloadBucket":
        if key is None:
            return cls("unknown", "unknown", "unknown")
        # Ignore legacy fine-grained query/composition suffixes. Runtime
        # calibration must be shared by phase-equivalent shapes; otherwise a
        # short serving run repeatedly re-explores M=2/4 and never reaches the
        # stable prefill-M>1/decode-M1 policy.
        padded = tuple(key) + ("unknown",) * (3 - len(key))
        return cls(
            str(padded[0]),
            str(padded[1]),
            str(padded[2]),
        )


@dataclass
class CalibrationState:
    count: int = 0
    ewma_error_ms: float = 0.0
    ewma_abs_error_ms: float = 0.0
    ewma_squared_error: float = 0.0
    ewma_log_ratio: float = 0.0
    ewma_squared_log_ratio: float = 0.0
    last_actual_ms: float | None = None
    last_predicted_ms: float | None = None
    last_prior_ms: float | None = None
    last_update_step: int = -1

    def update(
        self,
        *,
        error_ms: float,
        predicted_ms: float,
        prior_ms: float,
        actual_ms: float,
        step_id: int,
        alpha: float,
    ) -> None:
        log_ratio = math.log(max(actual_ms, 1e-9) / max(prior_ms, 1e-9))
        if self.count == 0:
            self.ewma_error_ms = error_ms
            self.ewma_abs_error_ms = abs(error_ms)
            self.ewma_squared_error = error_ms * error_ms
            self.ewma_log_ratio = log_ratio
            self.ewma_squared_log_ratio = log_ratio * log_ratio
        else:
            self.ewma_error_ms = (
                alpha * error_ms + (1.0 - alpha) * self.ewma_error_ms
            )
            self.ewma_abs_error_ms = (
                alpha * abs(error_ms)
                + (1.0 - alpha) * self.ewma_abs_error_ms
            )
            self.ewma_squared_error = (
                alpha * error_ms * error_ms
                + (1.0 - alpha) * self.ewma_squared_error
            )
            self.ewma_log_ratio = (
                alpha * log_ratio + (1.0 - alpha) * self.ewma_log_ratio
            )
            self.ewma_squared_log_ratio = (
                alpha * log_ratio * log_ratio
                + (1.0 - alpha) * self.ewma_squared_log_ratio
            )
        self.count += 1
        self.last_actual_ms = actual_ms
        self.last_predicted_ms = predicted_ms
        self.last_prior_ms = prior_ms
        self.last_update_step = step_id


@dataclass
class ContextualCostState:
    """Recursive least-squares model for a log-cost response."""

    dimension: int
    count: int = 0
    coefficients: np.ndarray = field(init=False)
    offline_coefficients: np.ndarray = field(init=False)
    covariance: np.ndarray = field(init=False)
    residual_variance: float = 0.0
    last_update_step: int = -1
    has_offline_prior: bool = False
    max_online_residual_ratio: float = 0.08

    def __post_init__(self) -> None:
        self.coefficients = np.zeros(self.dimension, dtype=np.float64)
        self.offline_coefficients = np.zeros(
            self.dimension, dtype=np.float64
        )
        self.covariance = np.eye(self.dimension, dtype=np.float64) * 4.0

    def _normalized_context(self, context: np.ndarray) -> np.ndarray:
        values = np.asarray(context, dtype=np.float64)
        if values.size == self.dimension:
            return values
        if values.size > self.dimension:
            return values[: self.dimension]
        return np.pad(values, (0, self.dimension - values.size))

    def predict_components(
        self, context: np.ndarray
    ) -> tuple[float, float, float]:
        context = self._normalized_context(context)
        raw_mean = float(context @ self.coefficients)
        offline_mean = float(context @ self.offline_coefficients)
        if self.has_offline_prior:
            residual_limit = math.log1p(
                max(0.0, self.max_online_residual_ratio)
            )
            online_residual = max(
                -residual_limit,
                min(residual_limit, raw_mean - offline_mean),
            )
            mean = offline_mean + online_residual
        else:
            online_residual = raw_mean
            mean = raw_mean
        return offline_mean, online_residual, mean

    def predict(self, context: np.ndarray) -> tuple[float, float]:
        context = self._normalized_context(context)
        _, _, mean = self.predict_components(context)
        leverage = max(0.0, float(context @ self.covariance @ context))
        residual_sigma = math.sqrt(max(0.0, self.residual_variance))
        # We decide using uncertainty in the expected response, not a
        # prediction interval for one noisy execution. Including the
        # irreducible ``+1`` term made a small, consistently beneficial arm
        # impossible to prove even after repeated observations.
        # Do not let RLS covariance collapse faster than the empirical
        # residual evidence. The former controller reported a narrow LCB even
        # while held-out step errors remained in the 10--25% range.
        empirical_mean_uncertainty = residual_sigma / math.sqrt(
            max(1, self.count)
        )
        uncertainty = max(
            residual_sigma * math.sqrt(leverage),
            empirical_mean_uncertainty,
        )
        return mean, uncertainty

    def update(
        self,
        *,
        context: np.ndarray,
        target: float,
        forgetting_factor: float,
        alpha: float,
        step_id: int,
    ) -> None:
        context = self._normalized_context(context)
        prediction = float(context @ self.coefficients)
        error = target - prediction
        covariance_context = self.covariance @ context
        denominator = max(
            1e-9,
            forgetting_factor + float(context @ covariance_context),
        )
        gain = covariance_context / denominator
        self.coefficients += gain * error
        self.covariance = (
            self.covariance
            - np.outer(gain, context) @ self.covariance
        ) / forgetting_factor
        squared_error = error * error
        if self.count == 0:
            self.residual_variance = squared_error
        else:
            self.residual_variance = (
                alpha * squared_error
                + (1.0 - alpha) * self.residual_variance
            )
        self.count += 1
        self.last_update_step = step_id


@dataclass(frozen=True)
class PendingContextualOutcome:
    bucket_key: tuple[str, str, str]
    selected_m: int
    context: tuple[float, ...]
    prior_ms: float
    actual_ms: float
    baseline_ms: float
    queue_depth: int | None
    waiting_reqs: int | None
    transition_sample: bool = False


@dataclass(frozen=True)
class PendingQueueOutcome:
    bucket_key: tuple[str, str, str]
    selected_m: int
    queue_depth: int | None
    waiting_reqs: int | None


@dataclass
class RegretWindowState:
    entries: deque[tuple[float, float]] = field(default_factory=deque)
    queue_bad_streak: int = 0
    safe_queue_growth: deque[float] = field(default_factory=deque)

    def add(
        self,
        *,
        regret_ms: float,
        baseline_ms: float,
        window_steps: int,
    ) -> None:
        self.entries.append((max(0.0, regret_ms), max(1e-6, baseline_ms)))
        while len(self.entries) > window_steps:
            self.entries.popleft()

    def regret_pct(self) -> float:
        if not self.entries:
            return 0.0
        regret = sum(entry[0] for entry in self.entries)
        baseline = sum(entry[1] for entry in self.entries)
        return regret / max(baseline, 1e-6) * 100.0

    def add_regret(
        self,
        *,
        regret_ms: float,
        baseline_ms: float,
        window_steps: int,
    ) -> None:
        """Charge an observed loss to its already-recorded decision step."""
        regret_ms = max(0.0, float(regret_ms))
        if self.entries:
            prior_regret, prior_baseline = self.entries.pop()
            self.entries.append(
                (prior_regret + regret_ms, prior_baseline)
            )
            return
        self.add(
            regret_ms=regret_ms,
            baseline_ms=baseline_ms,
            window_steps=window_steps,
        )

    def add_safe_queue_growth(
        self,
        *,
        growth: float,
        window_steps: int,
    ) -> None:
        self.safe_queue_growth.append(float(growth))
        while len(self.safe_queue_growth) > max(1, window_steps):
            self.safe_queue_growth.popleft()

    def safe_queue_growth_mean(self, *, min_samples: int) -> float:
        if len(self.safe_queue_growth) < max(1, min_samples):
            return 0.0
        return sum(self.safe_queue_growth) / len(self.safe_queue_growth)

    def safe_queue_positive_ratio(self, *, min_samples: int) -> float:
        if len(self.safe_queue_growth) < max(1, min_samples):
            return 0.0
        positive = sum(growth > 0.0 for growth in self.safe_queue_growth)
        return positive / len(self.safe_queue_growth)


@dataclass
class ContextRegimeState:
    ewma_context: tuple[float, ...] | None = None
    context_window: deque[tuple[float, ...]] = field(default_factory=deque)
    stable_observations: int = 0
    change_streak: int = 0
    warming_up: bool = False
    generation: int = 0
    generation_change_pending: bool = False
    exploration_attempts: dict[int, int] = field(default_factory=dict)
    probe_m: int | None = None
    probe_remaining: int = 0
    probe_generation: int = -1
    probe_context: tuple[float, ...] | None = None
    pending_probe_m: int | None = None
    pending_probe_generation: int = -1
    pending_probe_step: int = -1
    post_anchor_pending: bool = False
    safe_anchor_scale: float | None = None
    safe_anchor_context: tuple[float, ...] | None = None
    safe_anchor_step: int = -1
    safe_anchor_generation: int = -1


@dataclass
class ContextualValidationState:
    """State for a contiguous safe/candidate/safe serving experiment.

    A candidate is never promoted from scattered one-step samples.  Each
    stage is measured between two M=1 windows so arrival-rate drift is shared
    by the candidate and its counterfactual.
    """

    generation: int = -1
    target_bucket: tuple[str, str, str] | None = None
    candidate_m: int | None = None
    stage: int = 0
    phase: str = "idle"
    phase_step: int = 0
    window_id: int = 0
    experiment_id: int = 0
    safe_before_rate: float | None = None
    candidate_rate: float | None = None
    safe_before_target_share: float | None = None
    candidate_target_share: float | None = None
    safe_before_queue_growth: float = 0.0
    candidate_queue_growth: float = 0.0
    safe_before_age_growth_ms: float = 0.0
    candidate_age_growth_ms: float = 0.0
    candidate_context: tuple[float, ...] = ()
    phase_context_sum: np.ndarray | None = None
    phase_context_count: int = 0
    safe_before_context: tuple[float, ...] = ()
    safe_after_context: tuple[float, ...] = ()
    predicted_point_gain_pct: float | None = None
    predicted_gain_lcb_pct: float | None = None
    washout_remaining: int = 0
    last_target_step_id: int = -1


@dataclass
class ValidationEvidenceState:
    """Independent matched A/B/A evidence for one regime, arm and stage."""

    generation: int = -1
    stage: int = -1
    gains_pct: deque[float] = field(default_factory=deque)
    queue_safe: deque[bool] = field(default_factory=deque)
    contexts: deque[tuple[float, ...]] = field(default_factory=deque)

    def reset(self, *, generation: int, stage: int) -> None:
        self.generation = generation
        self.stage = stage
        self.gains_pct.clear()
        self.queue_safe.clear()
        self.contexts.clear()


@dataclass
class CandidateExposureState:
    """Legacy trace state retained for old trace compatibility."""

    generation: int = -1
    stage: int = 0
    credit: float = 0.0
    validation_ratios: deque[float] = field(default_factory=deque)
    validation_queue_growth: deque[float] = field(default_factory=deque)
    validated: bool = False


@dataclass
class BucketDecisionState:
    current_m: int
    last_change_step: int = -(10**9)
    pending_m: int | None = None
    pending_wins: int = 0
    bad_streak: int = 0


@dataclass(frozen=True)
class CandidateScore:
    m: int
    prior_cost_ms: float
    correction_ms: float
    calibrated_cost_ms: float
    uncertainty_ms: float
    robust_cost_ms: float
    count: int
    last_update_step: int = -1
    rejected: bool = False
    rejection_reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        calibration_scale = (
            self.calibrated_cost_ms / self.prior_cost_ms
            if self.prior_cost_ms > 0
            else None
        )
        return {
            "m": self.m,
            "prior_ms": self.prior_cost_ms,
            "correction_ms": self.correction_ms,
            "calibration_scale": calibration_scale,
            "calibrated_ms": self.calibrated_cost_ms,
            "uncertainty_ms": self.uncertainty_ms,
            "robust_ms": self.robust_cost_ms,
            "count": self.count,
            "last_update_step": self.last_update_step,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class AdaptiveUBatchDecision:
    num_ubatches: int
    predicted_gain_pct: float
    reason: str
    bucket_key: tuple[str, ...] | None = None
    total_tokens: int = 0
    online: bool = False
    num_reqs: int = 0
    predicted_cost_ms: float | None = None
    robust_cost_ms: float | None = None
    previous_m: int | None = None
    switched: bool = False
    fallback: bool = False
    decision_overhead_us: float | None = None
    candidate_scores: tuple[dict[str, Any], ...] = ()
    queue_depth: int | None = None
    waiting_reqs: int | None = None
    context_vector: tuple[float, ...] = ()
    contextual_baseline_ms: float | None = None
    contextual_gain_lcb_pct: float | None = None
    contextual_regret_pct: float = 0.0
    service_output_tokens: int = 0
    service_completed_reqs: int = 0
    max_waiting_age_ms: float = 0.0
    validation_window_id: int = -1
    validation_phase: str | None = None
    validation_boundary: bool = False
    validation_target_bucket: tuple[str, ...] | None = None
    validation_target_step: bool = False
    validation_stage: int = -1
    validation_experiment_id: int = -1

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_ubatches": self.num_ubatches,
            "predicted_gain_pct": self.predicted_gain_pct,
            "reason": self.reason,
            "bucket_key": self.bucket_key,
            "total_tokens": self.total_tokens,
            "online": self.online,
            "num_reqs": self.num_reqs,
            "predicted_cost_ms": self.predicted_cost_ms,
            "robust_cost_ms": self.robust_cost_ms,
            "previous_m": self.previous_m,
            "switched": self.switched,
            "fallback": self.fallback,
            "decision_overhead_us": self.decision_overhead_us,
            "candidate_scores": list(self.candidate_scores),
            "queue_depth": self.queue_depth,
            "waiting_reqs": self.waiting_reqs,
            "context_vector": list(self.context_vector),
            "contextual_baseline_ms": self.contextual_baseline_ms,
            "contextual_gain_lcb_pct": self.contextual_gain_lcb_pct,
            "contextual_regret_pct": self.contextual_regret_pct,
            "service_output_tokens": self.service_output_tokens,
            "service_completed_reqs": self.service_completed_reqs,
            "max_waiting_age_ms": self.max_waiting_age_ms,
            "validation_window_id": self.validation_window_id,
            "validation_phase": self.validation_phase,
            "validation_boundary": self.validation_boundary,
            "validation_target_bucket": self.validation_target_bucket,
            "validation_target_step": self.validation_target_step,
            "validation_stage": self.validation_stage,
            "validation_experiment_id": self.validation_experiment_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AdaptiveUBatchDecision":
        bucket_key = payload.get("bucket_key")
        if bucket_key is not None:
            bucket_key = tuple(bucket_key)
        validation_target_bucket = payload.get("validation_target_bucket")
        if validation_target_bucket is not None:
            validation_target_bucket = tuple(validation_target_bucket)
        return cls(
            num_ubatches=int(payload["num_ubatches"]),
            predicted_gain_pct=float(payload["predicted_gain_pct"]),
            reason=str(payload["reason"]),
            bucket_key=bucket_key,
            total_tokens=int(payload.get("total_tokens", 0)),
            online=bool(payload.get("online", False)),
            num_reqs=int(payload.get("num_reqs", 0)),
            predicted_cost_ms=_optional_float(payload.get("predicted_cost_ms")),
            robust_cost_ms=_optional_float(payload.get("robust_cost_ms")),
            previous_m=_optional_int(payload.get("previous_m")),
            switched=bool(payload.get("switched", False)),
            fallback=bool(payload.get("fallback", False)),
            decision_overhead_us=_optional_float(
                payload.get("decision_overhead_us")
            ),
            candidate_scores=tuple(payload.get("candidate_scores", ())),
            queue_depth=_optional_int(payload.get("queue_depth")),
            waiting_reqs=_optional_int(payload.get("waiting_reqs")),
            context_vector=tuple(
                float(value) for value in payload.get("context_vector", ())
            ),
            contextual_baseline_ms=_optional_float(
                payload.get("contextual_baseline_ms")
            ),
            contextual_gain_lcb_pct=_optional_float(
                payload.get("contextual_gain_lcb_pct")
            ),
            contextual_regret_pct=float(
                payload.get("contextual_regret_pct", 0.0)
            ),
            service_output_tokens=int(payload.get("service_output_tokens", 0)),
            service_completed_reqs=int(payload.get("service_completed_reqs", 0)),
            max_waiting_age_ms=float(payload.get("max_waiting_age_ms", 0.0)),
            validation_window_id=int(payload.get("validation_window_id", -1)),
            validation_phase=payload.get("validation_phase"),
            validation_boundary=bool(payload.get("validation_boundary", False)),
            validation_target_bucket=validation_target_bucket,
            validation_target_step=bool(
                payload.get("validation_target_step", False)
            ),
            validation_stage=int(payload.get("validation_stage", -1)),
            validation_experiment_id=int(
                payload.get("validation_experiment_id", -1)
            ),
        )


@dataclass(frozen=True)
class ExecutionObservation:
    selected_m: int
    bucket: WorkloadBucket
    predicted_cost_ms: float
    actual_step_ms: float
    success: bool = True
    failure_reason: str | None = None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _is_finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def _with_feature_counts(
    decision: AdaptiveUBatchDecision,
    features: AdaptiveUBatchFeatures,
) -> AdaptiveUBatchDecision:
    if (
        decision.total_tokens == features.total_tokens
        and decision.num_reqs == features.num_reqs
    ):
        return decision
    return AdaptiveUBatchDecision(
        num_ubatches=decision.num_ubatches,
        predicted_gain_pct=decision.predicted_gain_pct,
        reason=decision.reason,
        bucket_key=decision.bucket_key,
        total_tokens=features.total_tokens,
        online=decision.online,
        num_reqs=features.num_reqs,
        predicted_cost_ms=decision.predicted_cost_ms,
        robust_cost_ms=decision.robust_cost_ms,
        previous_m=decision.previous_m,
        switched=decision.switched,
        fallback=decision.fallback,
        decision_overhead_us=decision.decision_overhead_us,
        candidate_scores=decision.candidate_scores,
        queue_depth=decision.queue_depth,
        waiting_reqs=decision.waiting_reqs,
        context_vector=decision.context_vector,
        contextual_baseline_ms=decision.contextual_baseline_ms,
        contextual_gain_lcb_pct=decision.contextual_gain_lcb_pct,
        contextual_regret_pct=decision.contextual_regret_pct,
        service_output_tokens=decision.service_output_tokens,
        service_completed_reqs=decision.service_completed_reqs,
        max_waiting_age_ms=decision.max_waiting_age_ms,
        validation_window_id=decision.validation_window_id,
        validation_phase=decision.validation_phase,
        validation_boundary=decision.validation_boundary,
        validation_target_bucket=decision.validation_target_bucket,
        validation_target_step=decision.validation_target_step,
        validation_stage=decision.validation_stage,
        validation_experiment_id=decision.validation_experiment_id,
    )


def _extract_model_billions(model_config: Any) -> float | None:
    """Best-effort model-size extraction for local scheduling policy."""
    names = [
        getattr(model_config, "model", None),
        getattr(model_config, "served_model_name", None),
    ]
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        names.extend(
            [
                getattr(hf_config, "_name_or_path", None),
                getattr(hf_config, "model_type", None),
            ]
        )
    for name in names:
        if not name:
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*b", str(name), flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    if hf_config is None:
        return None
    hidden_size = int(getattr(hf_config, "hidden_size", 0) or 0)
    num_layers = int(getattr(hf_config, "num_hidden_layers", 0) or 0)
    if hidden_size >= 5000 or num_layers >= 48:
        return 14.0
    if hidden_size >= 3500:
        return 7.0
    if hidden_size > 0:
        return 3.0
    return None


def _extract_hidden_size(model_config: Any) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return 0
    hidden_size = int(getattr(hf_config, "hidden_size", 0) or 0)
    if hidden_size > 0:
        return hidden_size
    text_config = getattr(hf_config, "text_config", None)
    return int(getattr(text_config, "hidden_size", 0) or 0) if text_config else 0


def _cap_candidate(candidate: int, max_ubatches: int, total_tokens: int) -> int:
    return max(1, min(candidate, max(1, max_ubatches), max(1, total_tokens)))


def _scheduled_token_features(
    num_scheduled_tokens: np.ndarray | list[int],
) -> tuple[int, int, int, int, int, int, int, float, float, float, float]:
    tokens = np.asarray(num_scheduled_tokens, dtype=np.int64)
    if tokens.size == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0
    total = int(tokens.sum())
    max_query = int(tokens.max())
    num_reqs = int(tokens.size)
    prefill_tokens = int(np.maximum(tokens - 1, 0).sum())
    decode_tokens = max(0, total - prefill_tokens)
    prefill_reqs = int(np.count_nonzero(tokens > 1))
    decode_reqs = max(0, num_reqs - prefill_reqs)
    prefill_ratio = prefill_tokens / total if total > 0 else 0.0
    avg_tokens_per_req = total / num_reqs if num_reqs > 0 else 0.0
    token_imbalance = (
        float(np.std(tokens)) / max(avg_tokens_per_req, 1e-6)
        if num_reqs > 1
        else 0.0
    )
    smallest_request_ratio = (
        float(tokens.min()) / max(avg_tokens_per_req, 1e-6)
        if num_reqs > 0
        else 0.0
    )
    return (
        total,
        num_reqs,
        max_query,
        decode_tokens,
        prefill_tokens,
        prefill_reqs,
        decode_reqs,
        prefill_ratio,
        avg_tokens_per_req,
        token_imbalance,
        smallest_request_ratio,
    )


def _bucketize_features(
    *,
    model_b: float,
    total: int,
    max_query: int,
    prefill_reqs: int,
    decode_reqs: int,
    prefill_ratio: float,
) -> tuple[str, str, str]:
    # Coarse buckets are deliberate: the best controller pools calibration
    # across nearby model sizes instead of repeatedly re-calibrating short
    # serving runs.
    if model_b > 7.0:
        model_bucket = "large"
    elif model_b >= 3.0:
        model_bucket = "medium"
    else:
        model_bucket = "small"

    if prefill_ratio >= 0.70:
        phase_bucket = "prefill"
    elif prefill_ratio >= 0.10:
        phase_bucket = "mixed"
    else:
        phase_bucket = "decode"

    # Broad token bands intentionally match the original stable controller.
    # The analytical prior still sees exact token/query counts, while online
    # corrections are pooled so nearby prefill shapes do not each pay a fresh
    # live-traffic calibration budget.
    if total < 256:
        token_bucket = "small"
    elif total < 1024:
        token_bucket = "medium"
    else:
        token_bucket = "large"
    return model_bucket, phase_bucket, token_bucket


def _prefill_threshold_from_config(parallel_config: Any) -> float:
    threshold = getattr(
        parallel_config,
        "adaptive_ubatch_prefill_threshold_pct",
        0.0,
    )
    if threshold is None:
        threshold = 0.0
    return max(0.0, min(100.0, float(threshold))) / 100.0


def extract_adaptive_ubatch_features(
    *,
    model_config: Any,
    num_scheduled_tokens: np.ndarray | list[int],
) -> AdaptiveUBatchFeatures:
    (
        total,
        num_reqs,
        max_query,
        decode_tokens,
        prefill_tokens,
        prefill_reqs,
        decode_reqs,
        prefill_ratio,
        avg_tokens,
        token_imbalance,
        smallest_request_ratio,
    ) = (
        _scheduled_token_features(num_scheduled_tokens)
    )
    model_b = _extract_model_billions(model_config)
    model_b = model_b if model_b is not None else 7.0
    hidden_size = _extract_hidden_size(model_config)
    if hidden_size <= 0:
        if model_b >= 13.0:
            hidden_size = 5120
        elif model_b >= 6.0:
            hidden_size = 3584
        else:
            hidden_size = 2048
    return AdaptiveUBatchFeatures(
        total_tokens=total,
        num_reqs=num_reqs,
        max_query_len=max_query,
        decode_tokens=decode_tokens,
        prefill_tokens=prefill_tokens,
        prefill_reqs=prefill_reqs,
        decode_reqs=decode_reqs,
        prefill_ratio=prefill_ratio,
        avg_tokens_per_req=avg_tokens,
        token_imbalance=token_imbalance,
        smallest_request_ratio=smallest_request_ratio,
        model_billions=model_b,
        hidden_size=hidden_size,
        bucket_key=_bucketize_features(
            model_b=model_b,
            total=total,
            max_query=max_query,
            prefill_reqs=prefill_reqs,
            decode_reqs=decode_reqs,
            prefill_ratio=prefill_ratio,
        ),
    )


@dataclass(frozen=True)
class _AnalyticalParams:
    alpha: float
    beta: float
    gamma: float
    delta: float = 1.5
    epsilon: float = 1.2e-6
    sigma: float = 0.38


def _analytical_params_for_model(model_b: float) -> _AnalyticalParams:
    # Defaults are the fitted 310P PP=2 parameters summarized in the v2 plan.
    # They are used directly by the adaptive selector.
    if model_b >= 13.0:
        return _AnalyticalParams(
            alpha=0.73,
            beta=98.0,
            gamma=25.0,
            sigma=0.28,
        )
    if model_b >= 6.0:
        return _AnalyticalParams(
            alpha=0.52,
            beta=135.0,
            gamma=38.0,
            sigma=0.38,
        )
    return _AnalyticalParams(
        alpha=0.29,
        beta=124.0,
        gamma=42.0,
        sigma=0.52,
    )


def _effective_overlap(m: int, features: AdaptiveUBatchFeatures) -> float:
    if m <= 1:
        return 1.0
    if m == 2:
        base = 1.48
    elif m == 4:
        base = 1.78
    else:
        base = 1.0 + min(0.8, 0.28 * (m - 1))

    # Low-prefill batches do not have enough pipeline bubble to hide the split
    # overhead. This encodes the v2 prefill-threshold observation smoothly.
    if features.prefill_ratio < 0.80:
        base = min(base, 1.08)
    elif features.prefill_ratio < 0.90:
        base = min(base, 1.25 if m == 2 else 1.18)

    # Large-model M=4 overlap looks attractive in traces but often loses after
    # split/KV pressure; keep its analytical prior conservative.
    if features.model_billions >= 13.0 and m >= 4:
        base = min(base, 1.20)
    # The theoretical overlap is reachable only when there are enough
    # independent requests and their scheduled-token work is reasonably
    # balanced. Previously a three-request, highly skewed batch could receive
    # the same M=4 prior as a dense balanced batch, systematically overstating
    # the overlap benefit. These continuous factors are workload-derived, not
    # model/dataset-specific thresholds.
    fill = min(1.0, features.num_reqs / max(1.0, float(m)))
    balance = 1.0 / (1.0 + max(0.0, features.token_imbalance))
    smallest = max(0.0, min(1.0, features.smallest_request_ratio))
    # Keep this a screening prior rather than a hard exclusion: even a skewed
    # batch can expose some overlap after grouping, while a balanced batch can
    # retain the original analytical opportunity.
    realizable = (
        fill
        * (0.5 + 0.5 * balance)
        * (0.75 + 0.25 * smallest)
    )
    base = 1.0 + (base - 1.0) * realizable
    return max(1.0, min(float(m), base))


def _analytical_serial_cost_ms(
    m: int,
    features: AdaptiveUBatchFeatures,
) -> float:
    """Return modeled work before applying pipeline overlap."""
    params = _analytical_params_for_model(features.model_billions)
    total_tokens = max(1.0, float(features.total_tokens))
    decode_tokens = max(0.0, float(features.decode_tokens))
    m_float = max(1.0, float(m))
    compute = (
        params.alpha
        * total_tokens
        * (
            1.0
            + (m_float * params.beta)
            / (total_tokens + m_float * params.gamma)
        )
    )
    comm = (
        m_float * params.delta
        + params.epsilon * total_tokens * features.hidden_size
    )
    sample = params.sigma * decode_tokens
    return compute + comm + sample


def _analytical_cost_ms(
    m: int,
    features: AdaptiveUBatchFeatures,
) -> float:
    return _analytical_serial_cost_ms(m, features) / _effective_overlap(
        m,
        features,
    )


def _context_vector(
    features: AdaptiveUBatchFeatures,
    *,
    queue_depth: int | None,
    waiting_reqs: int | None,
) -> tuple[float, ...]:
    """Build a continuous workload and congestion context."""
    prefill_request_share = features.prefill_reqs / max(1, features.num_reqs)
    queue = max(0, int(queue_depth or 0))
    waiting = max(0, int(waiting_reqs or 0))
    return (
        1.0,
        math.log1p(max(0.0, features.model_billions))
        / CONTEXT_MODEL_LOG_NORMALIZER,
        math.log1p(max(0, features.total_tokens))
        / CONTEXT_TOKEN_LOG_NORMALIZER,
        math.log1p(max(0, features.num_reqs))
        / CONTEXT_REQUEST_LOG_NORMALIZER,
        math.log1p(max(0, features.max_query_len))
        / CONTEXT_TOKEN_LOG_NORMALIZER,
        math.log1p(max(0.0, features.avg_tokens_per_req))
        / CONTEXT_TOKEN_LOG_NORMALIZER,
        max(0.0, min(1.0, features.prefill_ratio)),
        max(0.0, min(1.0, prefill_request_share)),
        math.log1p(queue) / CONTEXT_REQUEST_LOG_NORMALIZER,
        math.log1p(waiting) / CONTEXT_REQUEST_LOG_NORMALIZER,
        min(1.0, waiting / max(1, queue)),
    )


def _relative_context_vector(
    context: tuple[float, ...] | np.ndarray,
    *,
    candidate_prior_ms: float,
    safe_prior_ms: float,
) -> np.ndarray:
    """Add the physics prior as a candidate-specific model feature."""
    base = np.asarray(context, dtype=np.float64)
    prior_log_ratio = math.log(
        max(float(candidate_prior_ms), 1e-9)
        / max(float(safe_prior_ms), 1e-9)
    )
    return np.concatenate((base, np.asarray((prior_log_ratio,))))


def _context_distance(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    if len(first) != len(second) or not first:
        return float("inf")
    return math.sqrt(
        sum((left - right) ** 2 for left, right in zip(first[1:], second[1:]))
        / max(1, len(first) - 1)
    )


def _candidate_ms(
    *,
    parallel_config: Any,
    features: AdaptiveUBatchFeatures,
) -> list[int]:
    max_ubatches = int(getattr(parallel_config, "adaptive_ubatch_max_size", 4) or 1)
    prefill_threshold = _prefill_threshold_from_config(parallel_config)
    min_tokens_m2 = int(
        getattr(parallel_config, "adaptive_ubatch_min_tokens_m2", 128) or 128
    )
    min_tokens_m4 = int(
        getattr(parallel_config, "adaptive_ubatch_min_tokens_m4", 512) or 512
    )
    min_prefill_m4 = float(
        getattr(parallel_config, "adaptive_ubatch_min_prefill_ratio_m4", 0.85)
    )
    candidates = [1]
    if features.prefill_ratio < prefill_threshold:
        return candidates
    if max_ubatches >= 2 and features.total_tokens >= min_tokens_m2:
        candidates.append(2)
    disable_m4_large = bool(
        getattr(parallel_config, "adaptive_ubatch_disable_m4_for_large_model", False)
    )
    allow_m4_large = not disable_m4_large or features.model_billions < 13.0
    if (
        max_ubatches >= 4
        and features.total_tokens >= min_tokens_m4
        and features.prefill_ratio >= min_prefill_m4
        and allow_m4_large
    ):
        candidates.append(4)
    return candidates


def _select_analytical_prior(
    *,
    parallel_config: Any,
    features: AdaptiveUBatchFeatures,
) -> AdaptiveUBatchDecision:
    min_gain = float(getattr(parallel_config, "adaptive_ubatch_min_gain_pct", 5.0))
    prefill_threshold = _prefill_threshold_from_config(parallel_config)

    if features.prefill_ratio < prefill_threshold:
        return AdaptiveUBatchDecision(
            1,
            0.0,
            (
                "analytical_prefill_below_threshold;"
                f"prefill={features.prefill_ratio:.3f}"
            ),
            features.bucket_key,
            features.total_tokens,
            predicted_cost_ms=_analytical_cost_ms(1, features),
            robust_cost_ms=_analytical_cost_ms(1, features),
        )

    candidates = _candidate_ms(parallel_config=parallel_config, features=features)

    if not candidates:
        candidates = [1]

    costs = {m: _analytical_cost_ms(m, features) for m in candidates}
    baseline = costs[1]
    best_m = min(costs, key=costs.get)
    predicted_gain = (
        (baseline - costs[best_m]) / max(baseline, 1e-9) * 100.0
    )

    if best_m <= 1 or predicted_gain < min_gain:
        return AdaptiveUBatchDecision(
            1,
            max(0.0, predicted_gain),
            (
                "analytical_gain_below_threshold;"
                f"best_m={best_m}; gain={predicted_gain:.2f}%"
            ),
            features.bucket_key,
            features.total_tokens,
            predicted_cost_ms=costs[1],
            robust_cost_ms=costs[1],
        )
    return AdaptiveUBatchDecision(
        _cap_candidate(
            best_m,
            int(getattr(parallel_config, "adaptive_ubatch_max_size", 4) or 1),
            features.total_tokens,
        ),
        predicted_gain,
        (
            "analytical_model_prior;"
            f"costs={','.join(f'M{m}:{costs[m]:.2f}' for m in sorted(costs))}"
        ),
        features.bucket_key,
        features.total_tokens,
        predicted_cost_ms=costs[best_m],
        robust_cost_ms=costs[best_m],
    )


def select_adaptive_ubatch_count(
    *,
    parallel_config: Any,
    model_config: Any,
    num_scheduled_tokens: np.ndarray | list[int],
) -> AdaptiveUBatchDecision:
    """Offline-prior selector used for cold start and compatibility."""

    max_ubatches = int(getattr(parallel_config, "adaptive_ubatch_max_size", 4) or 1)
    min_gain = float(getattr(parallel_config, "adaptive_ubatch_min_gain_pct", 5.0))
    features = extract_adaptive_ubatch_features(
        model_config=model_config,
        num_scheduled_tokens=num_scheduled_tokens,
    )
    if features.total_tokens <= 1 or features.num_reqs <= 1:
        return AdaptiveUBatchDecision(
            1,
            0.0,
            "too_few_tokens_or_requests",
            features.bucket_key,
            features.total_tokens,
        )

    if bool(getattr(parallel_config, "adaptive_ubatch_use_analytical_prior", True)):
        return _select_analytical_prior(
            parallel_config=parallel_config,
            features=features,
        )

    candidate = 1
    predicted_gain = 0.0
    reason = "predicted_gain_below_threshold"

    if features.model_billions >= 13.0:
        if (
            features.prefill_ratio >= 0.95
            and features.total_tokens >= 384
            and max_ubatches >= 2
        ):
            candidate = 2
            predicted_gain = 2.0
            reason = "14b_extreme_prefill_guarded_m2"
        else:
            reason = "14b_split_overhead_guard"
    elif features.model_billions >= 6.0:
        if (
            features.prefill_ratio >= 0.90
            and features.max_query_len >= 512
            and features.total_tokens >= 256
        ):
            candidate = 4
            predicted_gain = 18.0
            reason = "7b_long_prefill_m4"
        elif features.prefill_ratio >= 0.65 and features.total_tokens >= 128:
            candidate = 2
            predicted_gain = 10.0
            reason = "7b_prefill_heavy_m2"
        elif features.total_tokens >= 128:
            candidate = 2
            predicted_gain = 6.0
            reason = "7b_large_batch_m2"
    else:
        if features.total_tokens >= 64:
            candidate = 2
            predicted_gain = 7.0
            reason = "3b_general_m2"

    if predicted_gain < min_gain:
        return AdaptiveUBatchDecision(
            1,
            predicted_gain,
            reason,
            features.bucket_key,
            features.total_tokens,
        )
    candidate = _cap_candidate(candidate, max_ubatches, features.total_tokens)
    if candidate <= 1:
        return AdaptiveUBatchDecision(
            1,
            predicted_gain,
            "candidate_capped_to_one",
            features.bucket_key,
            features.total_tokens,
        )
    return AdaptiveUBatchDecision(
        candidate,
        predicted_gain,
        reason,
        features.bucket_key,
        features.total_tokens,
    )


class AdaptiveUBatchController:
    """Adaptive micro-batch controller with a contextual safety layer.

    The calibrated analytical policy proposes an M. Contextual relative-cost
    estimates, a rolling regret budget versus safe M, workload-change
    detection, and multi-step queue drift then decide whether to execute it.
    """

    def __init__(
        self,
        *,
        parallel_config: Any,
        model_config: Any,
        trace_enabled: bool = True,
    ) -> None:
        self.parallel_config = parallel_config
        self.model_config = model_config
        self._lock = RLock()
        self._step_id = 0
        self._current_m = max(1, int(getattr(parallel_config, "adaptive_ubatch_safe_m", 1)))
        self._last_change_step = -10**9
        self._last_explore_step = -10**9
        self._last_bucket: WorkloadBucket | None = None
        self._stable_bucket_steps = 0
        self._bucket_decisions: dict[
            tuple[str, str, str], BucketDecisionState
        ] = {}
        self._calibration: dict[
            tuple[tuple[str, str, str], int],
            CalibrationState,
        ] = {}
        self._contextual_cost: dict[int, ContextualCostState] = {}
        self._regret_windows: dict[
            tuple[str, str, str], RegretWindowState
        ] = {}
        # The bucket-wide window is the hard serving-safety budget.  These
        # per-arm windows prevent one bad M from being treated as evidence
        # against every other candidate after the global window has recovered.
        self._arm_regret_windows: dict[
            tuple[tuple[str, str, str], int], RegretWindowState
        ] = {}
        self._context_regimes: dict[
            tuple[str, str, str], ContextRegimeState
        ] = {}
        self._candidate_exposure: dict[
            tuple[tuple[str, str, str], int], CandidateExposureState
        ] = {}
        self._validation_evidence: dict[
            tuple[tuple[str, str, str], int], ValidationEvidenceState
        ] = {}
        # Only one live A/B/A experiment may run at a time.  Other workload
        # buckets remain M=1 controls but are part of the same wall-clock
        # service window, so their work and queue effects are not discarded.
        self._active_validation = ContextualValidationState()
        self._next_validation_window_id = 0
        self._next_validation_experiment_id = 0
        self._last_validation_meta: tuple[int, str | None, bool] = (
            -1,
            None,
            False,
        )
        self._last_validation_target_bucket: tuple[str, str, str] | None = None
        self._last_validation_target_step = False
        self._last_validation_stage = -1
        self._last_validation_experiment_id = -1
        self._pending_contextual_outcome: PendingContextualOutcome | None = None
        self._pending_queue_outcome: PendingQueueOutcome | None = None
        self._cooldown_until: dict[
            tuple[tuple[str, str, str], int],
            int,
        ] = {}
        # The first timed execution of a new (bucket, M) commonly includes
        # shape compilation. Keep it out of the online cost model so one cold
        # sample cannot permanently eliminate the safe candidate.
        self._discarded_cold_samples: set[
            tuple[tuple[str, str, str], int]
        ] = set()
        configured_trace_path = getattr(
            parallel_config, "adaptive_ubatch_trace_path", None
        ) or os.getenv("VLLM_ADAPTIVE_UBATCH_TRACE_PATH")
        self._trace_path = (
            configured_trace_path if trace_enabled else None
        )
        self._trace_file: Any | None = None
        self._trace_records_since_flush = 0
        if self._trace_path:
            atexit.register(self.close_trace)

    def _mode(self) -> str:
        return str(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_mode",
                "contextual_safe",
            )
        )

    def _min_observations(self) -> int:
        explicit = getattr(
            self.parallel_config,
            "adaptive_ubatch_min_observations",
            None,
        )
        if explicit is not None:
            return max(1, int(explicit))
        return max(
            1,
            int(getattr(self.parallel_config, "adaptive_ubatch_warmup_steps", 8) or 8),
        )

    def _alpha(self) -> float:
        return max(
            1e-6,
            min(1.0, float(getattr(self.parallel_config, "adaptive_ubatch_ewma_alpha", 0.2))),
        )

    def _state(self, bucket: WorkloadBucket, m: int) -> CalibrationState:
        key = (bucket.as_tuple(), m)
        state = self._calibration.get(key)
        if state is None:
            state = CalibrationState()
            self._calibration[key] = state
        return state

    def _contextual_state(self, m: int) -> ContextualCostState:
        state = self._contextual_cost.get(m)
        if state is None:
            state = ContextualCostState(dimension=CONTEXT_DIMENSION)
            offline_prior = _OFFLINE_CONTEXTUAL_PRIORS.get(int(m))
            if offline_prior is not None:
                coefficients, residual_sigma = offline_prior
                state.offline_coefficients = np.asarray(
                    coefficients,
                    dtype=np.float64,
                )
                state.coefficients = state.offline_coefficients.copy()
                state.covariance = (
                    np.eye(CONTEXT_DIMENSION, dtype=np.float64) * 0.25
                )
                state.residual_variance = residual_sigma * residual_sigma
                state.has_offline_prior = True
                state.max_online_residual_ratio = max(
                    0.0,
                    float(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_online_residual_limit_pct",
                            8.0,
                        )
                    )
                    / 100.0,
                )
            self._contextual_cost[m] = state
        return state

    def _validation_evidence_state(
        self,
        bucket: WorkloadBucket,
        m: int,
        stage: int,
    ) -> ValidationEvidenceState:
        key = (bucket.as_tuple(), max(1, int(m)))
        state = self._validation_evidence.get(key)
        if state is None:
            state = ValidationEvidenceState()
            self._validation_evidence[key] = state
        generation = self._regime_state(bucket).generation
        if state.generation != generation or state.stage != stage:
            state.reset(generation=generation, stage=stage)
        return state

    def _regret_state(self, bucket: WorkloadBucket) -> RegretWindowState:
        key = bucket.as_tuple()
        state = self._regret_windows.get(key)
        if state is None:
            state = RegretWindowState()
            self._regret_windows[key] = state
        return state

    def _arm_regret_state(
        self,
        bucket: WorkloadBucket,
        m: int,
    ) -> RegretWindowState:
        key = (bucket.as_tuple(), max(1, int(m)))
        state = self._arm_regret_windows.get(key)
        if state is None:
            state = RegretWindowState()
            self._arm_regret_windows[key] = state
        return state

    def _exposure_state(
        self,
        bucket: WorkloadBucket,
        m: int,
    ) -> CandidateExposureState:
        key = (bucket.as_tuple(), max(1, int(m)))
        state = self._candidate_exposure.get(key)
        if state is None:
            state = CandidateExposureState()
            self._candidate_exposure[key] = state
        regime_generation = self._regime_state(bucket).generation
        if state.generation != regime_generation:
            state.generation = regime_generation
            state.stage = 0
            state.credit = 0.0
            state.validation_ratios.clear()
            state.validation_queue_growth.clear()
            state.validated = False
        return state

    def _exposure_ratios(self) -> tuple[float, ...]:
        configured = getattr(
            self.parallel_config,
            "adaptive_ubatch_exposure_stages",
            (0.05, 0.10, 0.25, 0.50),
        )
        if isinstance(configured, str):
            values = tuple(
                float(value.strip())
                for value in configured.split(",")
                if value.strip()
            )
        else:
            values = tuple(float(value) for value in configured)
        normalized = tuple(
            sorted({max(1e-3, min(1.0, value)) for value in values})
        )
        return normalized or (0.05, 0.10, 0.25, 0.50)

    def _record_exposure_observation(
        self,
        *,
        bucket: WorkloadBucket,
        m: int,
        relative_ratio: float,
        queue_growth: float | None,
    ) -> tuple[int, str | None]:
        state = self._exposure_state(bucket, m)
        if state.stage <= 0:
            # Initial probes are the evidence for entering the first bounded
            # exposure stage.
            state.stage = 1
        state.validation_ratios.append(max(1e-6, float(relative_ratio)))
        if queue_growth is not None:
            state.validation_queue_growth.append(float(queue_growth))
        required = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exposure_validation_observations",
                    3,
                )
            ),
        )
        if len(state.validation_ratios) < required:
            return state.stage, None
        ratios = sorted(state.validation_ratios)
        median_ratio = ratios[len(ratios) // 2]
        queue_positive_ratio = (
            sum(value > 0.0 for value in state.validation_queue_growth)
            / len(state.validation_queue_growth)
            if state.validation_queue_growth
            else 0.0
        )
        min_gain = max(
            0.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exposure_promotion_gain_pct",
                    2.0,
                )
            ),
        )
        queue_limit = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_exposure_queue_positive_ratio",
                        0.75,
                    )
                ),
            ),
        )
        stages = self._exposure_ratios()
        if (
            median_ratio <= 1.0 - min_gain / 100.0
            and queue_positive_ratio <= queue_limit
        ):
            if state.stage < len(stages):
                state.stage += 1
                state.validated = False
                outcome = "promoted"
            else:
                state.validated = True
                outcome = "validated"
        else:
            state.stage = max(0, state.stage - 1)
            state.validated = False
            state.credit = 0.0
            outcome = "demoted"
        state.validation_ratios.clear()
        state.validation_queue_growth.clear()
        return state.stage, outcome

    def _validation_stage_steps(self) -> tuple[int, ...]:
        configured = getattr(
            self.parallel_config,
            "adaptive_ubatch_validation_stage_steps",
            "4,8,16,32",
        )
        values = (
            tuple(int(value.strip()) for value in configured.split(",") if value.strip())
            if isinstance(configured, str)
            else tuple(int(value) for value in configured)
        )
        normalized = tuple(sorted({max(1, value) for value in values}))
        return normalized or (4, 8, 16, 32)

    def _validation_state(
        self,
        bucket: WorkloadBucket,
    ) -> ContextualValidationState:
        state = self._active_validation
        target = bucket.as_tuple()
        if state.phase == "idle":
            return state
        if state.target_bucket == target:
            generation = self._regime_state(bucket).generation
            if state.generation != generation:
                self._cancel_validation("context_generation_changed")
                state = self._active_validation
        return state

    def _cancel_validation(self, reason: str) -> None:
        state = self._active_validation
        if state.phase != "idle":
            self._write_trace({
                "type": "adaptive_ubatch_validation_cancelled",
                "step_id": self._step_id,
                "target_bucket": state.target_bucket,
                "candidate_m": state.candidate_m,
                "experiment_id": state.experiment_id,
                "stage": state.stage,
                "phase": state.phase,
                "reason": reason,
            })
        self._active_validation = ContextualValidationState()

    def _set_validation_meta(
        self,
        state: ContextualValidationState,
        *,
        boundary: bool,
        target_step: bool,
    ) -> None:
        self._last_validation_meta = (
            state.window_id,
            state.phase,
            boundary,
        )
        self._last_validation_target_bucket = state.target_bucket
        self._last_validation_target_step = target_step
        self._last_validation_stage = state.stage
        self._last_validation_experiment_id = state.experiment_id

    def _record_validation_context(
        self,
        state: ContextualValidationState,
        context: tuple[float, ...],
    ) -> None:
        values = np.asarray(context, dtype=np.float64)
        if state.phase_context_sum is None:
            state.phase_context_sum = values.copy()
        else:
            state.phase_context_sum += values
        state.phase_context_count += 1

    def _finish_validation_phase_context(
        self,
        state: ContextualValidationState,
    ) -> tuple[float, ...]:
        if state.phase_context_sum is None or state.phase_context_count <= 0:
            return ()
        result = tuple(
            float(value)
            for value in state.phase_context_sum / state.phase_context_count
        )
        state.phase_context_sum = None
        state.phase_context_count = 0
        return result

    def _validation_phase_steps(
        self,
        state: ContextualValidationState,
    ) -> int:
        stages = self._validation_stage_steps()
        candidate_steps = stages[min(state.stage, len(stages) - 1)]
        if state.phase == "candidate":
            return candidate_steps
        return max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_validation_safe_steps",
                    8,
                )
            ),
        )

    def _select_validation_phase(
        self,
        *,
        state: ContextualValidationState,
        safe_score: CandidateScore,
        candidate_scores: list[CandidateScore],
        context: tuple[float, ...],
    ) -> tuple[CandidateScore, str] | None:
        if state.phase == "idle" or state.candidate_m is None:
            return None
        candidate = next(
            (
                score
                for score in candidate_scores
                if score.m == state.candidate_m and not score.rejected
            ),
            None,
        )
        if candidate is None:
            self._cancel_validation("candidate_became_ineligible")
            return None
        selected = candidate if state.phase == "candidate" else safe_score
        state.last_target_step_id = self._step_id
        if state.washout_remaining > 0:
            state.washout_remaining -= 1
            return selected, f"contextual_validation_{state.phase}_washout"
        state.phase_step += 1
        relative_context = _relative_context_vector(
            context,
            candidate_prior_ms=candidate.prior_cost_ms,
            safe_prior_ms=safe_score.prior_cost_ms,
        )
        self._record_validation_context(
            state,
            tuple(float(value) for value in relative_context),
        )
        boundary = state.phase_step >= self._validation_phase_steps(state)
        self._set_validation_meta(
            state,
            boundary=boundary,
            target_step=True,
        )
        return selected, f"contextual_validation_{state.phase}"

    def _safe_anchor_is_valid(
        self,
        *,
        regime: ContextRegimeState,
        context: tuple[float, ...],
    ) -> bool:
        if (
            regime.safe_anchor_scale is None
            or regime.safe_anchor_context is None
            or regime.safe_anchor_generation != regime.generation
        ):
            return False
        stable_steps = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exploration_stable_steps",
                    8,
                )
            ),
        )
        regret_window_steps = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_regret_window_steps",
                    64,
                )
            ),
        )
        if self._step_id - regime.safe_anchor_step > max(
            4,
            stable_steps * 2,
            regret_window_steps,
        ):
            return False
        distance_limit = max(
            1e-6,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_context_change_threshold",
                    0.12,
                )
            ),
        )
        return (
            _context_distance(regime.safe_anchor_context, context)
            <= distance_limit
        )

    @staticmethod
    def _clear_probe(regime: ContextRegimeState) -> None:
        regime.probe_m = None
        regime.probe_remaining = 0
        regime.probe_generation = -1
        regime.probe_context = None

    def _regime_state(self, bucket: WorkloadBucket) -> ContextRegimeState:
        key = bucket.as_tuple()
        state = self._context_regimes.get(key)
        if state is None:
            state = ContextRegimeState()
            self._context_regimes[key] = state
        return state

    def _decision_state(self, bucket: WorkloadBucket) -> BucketDecisionState:
        key = bucket.as_tuple()
        state = self._bucket_decisions.get(key)
        if state is None:
            state = BucketDecisionState(
                current_m=max(
                    1,
                    int(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_safe_m",
                            1,
                        )
                    ),
                )
            )
            self._bucket_decisions[key] = state
        return state

    def _lookup_state(self, bucket: WorkloadBucket, m: int) -> CalibrationState:
        exact = self._calibration.get((bucket.as_tuple(), m))
        if exact is not None and exact.count > 0:
            return exact
        return CalibrationState()

    def _calibration_scale(self, state: CalibrationState) -> float:
        if state.count <= 0:
            return 1.0
        max_scale = max(
            1.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_max_calibration_scale",
                    8.0,
                )
            ),
        )
        scale = math.exp(
            max(-math.log(max_scale), min(math.log(max_scale), state.ewma_log_ratio))
        )
        return max(1.0 / max_scale, min(max_scale, scale))

    def _uncertainty_ms(self, calibrated: float, state: CalibrationState) -> float:
        log_variance = (
            state.ewma_squared_log_ratio
            - state.ewma_log_ratio * state.ewma_log_ratio
        )
        log_sigma = math.sqrt(max(log_variance, 0.0))
        max_scale = max(
            1.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_max_calibration_scale",
                    8.0,
                )
            ),
        )
        uncertainty = calibrated * (
            math.exp(min(log_sigma, math.log(max_scale))) - 1.0
        )
        min_obs = self._min_observations()
        if state.count < min_obs:
            penalty_ratio = float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_cold_start_penalty_ratio",
                    0.15,
                )
            )
            scarcity = 1.0 - state.count / max(1, min_obs)
            uncertainty += calibrated * penalty_ratio * max(0.0, scarcity)
        return uncertainty

    def _predict_safe_contextual_cost(
        self,
        *,
        score: CandidateScore,
        context: np.ndarray,
    ) -> tuple[float, float, int]:
        state = self._contextual_state(score.m)
        safe_context = np.concatenate((context, np.asarray((0.0,))))
        log_scale, uncertainty = state.predict(safe_context)
        max_scale = max(
            1.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_max_calibration_scale",
                    8.0,
                )
            ),
        )
        log_limit = math.log(max_scale)
        log_scale = max(-log_limit, min(log_limit, log_scale))
        min_observations = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_context_min_observations",
                    3,
                )
            ),
        )
        if state.count < min_observations:
            scarcity = 1.0 - state.count / min_observations
            uncertainty += float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_cold_start_penalty_ratio",
                    0.15,
                )
            ) * scarcity
        cost_ms = max(1e-6, score.prior_cost_ms * math.exp(log_scale))
        return cost_ms, max(0.0, uncertainty), state.count

    def _predict_relative_candidate(
        self,
        *,
        score: CandidateScore,
        safe_score: CandidateScore,
        context: np.ndarray,
    ) -> tuple[float, float, int]:
        """Predict T(M) / T(safe M) directly in log space."""
        state = self._contextual_state(score.m)
        relative_context = _relative_context_vector(
            context,
            candidate_prior_ms=score.prior_cost_ms,
            safe_prior_ms=safe_score.prior_cost_ms,
        )
        log_ratio, log_uncertainty = state.predict(relative_context)
        max_scale = max(
            1.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_max_calibration_scale",
                    8.0,
                )
            ),
        )
        log_limit = math.log(max_scale)
        if state.count == 0 and state.has_offline_prior:
            # Do not extrapolate a fitted cold-start prior beyond the range
            # supported by held-out matched windows. Live evidence can move
            # beyond this bound after the first accepted observation.
            log_limit = min(log_limit, math.log(1.35))
        log_ratio = max(-log_limit, min(log_limit, log_ratio))
        relative_ratio = math.exp(log_ratio)
        min_observations = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_context_min_observations",
                    3,
                )
            ),
        )
        if state.count < min_observations and not state.has_offline_prior:
            scarcity = 1.0 - state.count / min_observations
            log_uncertainty += float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_cold_start_penalty_ratio",
                    0.15,
                )
            ) * scarcity
        return (
            max(1e-6, relative_ratio),
            max(0.0, log_uncertainty),
            state.count,
        )

    def _update_context_regime(
        self,
        *,
        bucket: WorkloadBucket,
        context: tuple[float, ...],
    ) -> tuple[bool, float]:
        state = self._regime_state(bucket)
        stable_required = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exploration_stable_steps",
                    8,
                )
            ),
        )
        confirmations = max(
            2,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_switch_confirmations",
                    2,
                )
            ),
        )
        window_size = max(stable_required, confirmations * 2)
        state.context_window.append(context)
        while len(state.context_window) > window_size:
            state.context_window.popleft()
        if state.ewma_context is None:
            state.ewma_context = context
            state.stable_observations = 1
            return False, 0.0
        window_context = tuple(
            float(
                np.median(
                    [sample[index] for sample in state.context_window]
                )
            )
            for index in range(len(context))
        )
        distance = _context_distance(
            state.ewma_context,
            window_context,
        )
        instant_distance = _context_distance(
            state.ewma_context,
            context,
        )
        safety_distance = max(distance, instant_distance)
        threshold = float(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_context_change_threshold",
                0.12,
            )
        )
        alpha = self._alpha()
        if len(state.context_window) < window_size:
            state.ewma_context = tuple(
                alpha * current + (1.0 - alpha) * previous
                for previous, current in zip(
                    state.ewma_context,
                    window_context,
                )
            )
            state.stable_observations += 1
            return (
                state.warming_up or instant_distance > threshold,
                safety_distance,
            )
        if distance > threshold:
            state.change_streak += 1
            # Stability is local to this contextual bucket.  An intervening
            # decode/other bucket must not erase prior stable visits, but a
            # material change within this bucket must make it safe again.
            state.stable_observations = 0
            if not state.warming_up:
                state.generation_change_pending = True
            state.warming_up = True
            if state.change_streak >= confirmations:
                if state.generation_change_pending:
                    state.generation += 1
                    state.exploration_attempts.clear()
                    self._clear_probe(state)
                    state.pending_probe_m = None
                    state.pending_probe_generation = -1
                    state.pending_probe_step = -1
                    state.post_anchor_pending = False
                    state.safe_anchor_scale = None
                    state.safe_anchor_context = None
                    state.safe_anchor_step = -1
                    state.safe_anchor_generation = -1
                    state.generation_change_pending = False
                state.ewma_context = window_context
                state.stable_observations = 1
                state.change_streak = 0
        else:
            state.change_streak = 0
            state.ewma_context = tuple(
                alpha * current + (1.0 - alpha) * previous
                for previous, current in zip(
                    state.ewma_context,
                    window_context,
                )
            )
            state.stable_observations += 1
            if state.stable_observations >= stable_required:
                state.warming_up = False
                state.generation_change_pending = False
        return (
            state.warming_up or instant_distance > threshold,
            safety_distance,
        )

    def _finalize_queue_outcome(
        self,
        *,
        queue_depth: int | None,
        waiting_reqs: int | None,
    ) -> None:
        """Update congestion trend from scheduler state without timing sync."""
        pending = self._pending_queue_outcome
        self._pending_queue_outcome = None
        if (
            pending is None
            or queue_depth is None
            or pending.queue_depth is None
        ):
            return
        safe_m = max(
            1,
            int(getattr(self.parallel_config, "adaptive_ubatch_safe_m", 1)),
        )
        if pending.selected_m != safe_m:
            return
        stable_steps = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exploration_stable_steps",
                    8,
                )
            ),
        )
        self._regret_state(
            WorkloadBucket.from_key(pending.bucket_key)
        ).add_safe_queue_growth(
            growth=queue_depth - pending.queue_depth,
            window_steps=max(8, stable_steps * 4),
        )

    def _remember_queue_outcome(
        self,
        decision: AdaptiveUBatchDecision,
    ) -> None:
        if self._mode() != "contextual_safe":
            return
        self._pending_queue_outcome = PendingQueueOutcome(
            bucket_key=WorkloadBucket.from_key(
                decision.bucket_key
            ).as_tuple(),
            selected_m=max(1, int(decision.num_ubatches)),
            queue_depth=decision.queue_depth,
            waiting_reqs=decision.waiting_reqs,
        )

    def _finalize_contextual_outcome(
        self,
        *,
        queue_depth: int | None,
        waiting_reqs: int | None,
    ) -> None:
        pending = self._pending_contextual_outcome
        self._pending_contextual_outcome = None
        if pending is None:
            return
        context = np.asarray(pending.context, dtype=np.float64)
        state = self._contextual_state(pending.selected_m)
        safe_m = max(
            1,
            int(getattr(self.parallel_config, "adaptive_ubatch_safe_m", 1)),
        )
        reference_ms = (
            pending.prior_ms
            if pending.selected_m == safe_m
            else pending.baseline_ms
        )
        raw_target = math.log(
            max(pending.actual_ms, 1e-9) / max(reference_ms, 1e-9)
        )
        # Full-worker feedback includes queueing and pipeline jitter that is
        # not caused by M alone.  Bound each recursive-model innovation so a
        # single noisy step cannot reverse an otherwise consistent arm.
        predicted_target, _ = state.predict(context)
        max_correction_ratio = max(
            0.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_max_correction_ratio",
                    0.3,
                )
            ),
        )
        innovation_limit = math.log1p(max_correction_ratio)
        innovation = raw_target - predicted_target
        target = predicted_target + max(
            -innovation_limit,
            min(innovation_limit, innovation),
        )
        # M>1 full-worker observations have no simultaneous M=1
        # counterfactual.  Updating the relative model from them previously
        # let queue and pipeline jitter overwrite the offline prior.  The
        # candidate model is now updated only by matched A/B/A evidence in
        # observe_service_window; ordinary feedback only maintains M=1.
        if pending.selected_m == safe_m and not pending.transition_sample:
            state.update(
                context=context,
                target=target,
                forgetting_factor=float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_context_forgetting_factor",
                        0.98,
                    )
                ),
                alpha=self._alpha(),
                step_id=self._step_id,
            )
        bucket = WorkloadBucket.from_key(pending.bucket_key)
        regret_state = self._regret_state(bucket)
        regime_state = self._regime_state(bucket)
        if pending.selected_m == safe_m and not pending.transition_sample:
            max_scale = max(
                1.0,
                float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_max_calibration_scale",
                        8.0,
                    )
                ),
            )
            measured_scale = pending.actual_ms / max(pending.prior_ms, 1e-6)
            regime_state.safe_anchor_scale = max(
                1.0 / max_scale,
                min(max_scale, measured_scale),
            )
            regime_state.safe_anchor_context = pending.context
            regime_state.safe_anchor_step = self._step_id
            regime_state.safe_anchor_generation = regime_state.generation
        raw_regret_ms = (
            max(0.0, pending.actual_ms - pending.baseline_ms)
            if pending.selected_m != safe_m
            else 0.0
        )
        max_exploration_regret_pct = max(
            0.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_max_exploration_regret_pct",
                    5.0,
                )
            ),
        )
        # A transition can be excluded from the steady-state cost model, but
        # the serving system still paid its complete latency cost. Charging
        # the full loss is necessary for the 2% conservative budget to mean
        # anything at the end-to-end level.
        regret_ms = raw_regret_ms
        window_steps = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_regret_window_steps",
                    64,
                )
            ),
        )
        regret_state.add_regret(
            regret_ms=regret_ms,
            baseline_ms=pending.baseline_ms,
            window_steps=window_steps,
        )
        if pending.selected_m != safe_m:
            self._arm_regret_state(
                bucket,
                pending.selected_m,
            ).add_regret(
                regret_ms=regret_ms,
                baseline_ms=pending.baseline_ms,
                window_steps=window_steps,
            )
        candidate_cooldown_until = self._cooldown_until.get(
            (bucket.as_tuple(), pending.selected_m),
            -1,
        )
        raw_regret_pct = (
            raw_regret_ms / max(pending.baseline_ms, 1e-6) * 100.0
        )
        probe_state = self._regime_state(bucket)
        probe_active = (
            probe_state.probe_m == pending.selected_m
            and probe_state.probe_remaining > 0
            and probe_state.probe_generation == probe_state.generation
        )
        if (
            pending.selected_m != safe_m
            and raw_regret_pct > max_exploration_regret_pct
            and not pending.transition_sample
            and not probe_active
        ):
            candidate_cooldown_until = self._step_id + max(
                1,
                int(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_failure_cooldown_steps",
                        32,
                    )
                ),
            )
            self._cooldown_until[
                (bucket.as_tuple(), pending.selected_m)
            ] = candidate_cooldown_until
        queue_growth = (
            queue_depth - pending.queue_depth
            if queue_depth is not None and pending.queue_depth is not None
            else None
        )
        waiting_growth = (
            waiting_reqs - pending.waiting_reqs
            if waiting_reqs is not None and pending.waiting_reqs is not None
            else None
        )
        queue_window_steps = max(
            8,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exploration_stable_steps",
                    8,
                )
            ) * 4,
        )
        safe_queue_growth_mean = regret_state.safe_queue_growth_mean(
            min_samples=min(8, queue_window_steps),
        )
        safe_queue_positive_ratio = regret_state.safe_queue_positive_ratio(
            min_samples=min(8, queue_window_steps),
        )
        threshold = max(
            0,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_queue_growth_threshold",
                    2,
                )
            ),
        )
        queue_regressed = (
            pending.selected_m != safe_m
            and bool(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_queue_safety_enabled",
                    True,
                )
            )
            and queue_growth is not None
            and waiting_growth is not None
            and queue_growth > threshold
            and waiting_growth > threshold
        )
        regret_state.queue_bad_streak = (
            regret_state.queue_bad_streak + 1
            if queue_regressed
            else max(0, regret_state.queue_bad_streak - 1)
        )
        confirmations = max(
            2,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_switch_confirmations",
                    2,
                )
            ),
        )
        if regret_state.queue_bad_streak >= confirmations:
            self._cooldown_until[(bucket.as_tuple(), pending.selected_m)] = (
                self._step_id
                + max(
                    1,
                    int(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_failure_cooldown_steps",
                            32,
                        )
                    ),
                )
            )
            if pending.selected_m != safe_m:
                self._clear_probe(regime_state)
                regime_state.pending_probe_m = None
                regime_state.pending_probe_generation = -1
                regime_state.pending_probe_step = -1
                regime_state.post_anchor_pending = True
            regret_state.queue_bad_streak = 0
        exposure_stage = None
        exposure_outcome = None
        if (
            pending.selected_m != safe_m
            and not pending.transition_sample
        ):
            exposure_stage, exposure_outcome = (
                self._record_exposure_observation(
                    bucket=bucket,
                    m=pending.selected_m,
                    relative_ratio=(
                        pending.actual_ms
                        / max(pending.baseline_ms, 1e-9)
                    ),
                    queue_growth=queue_growth,
                )
            )
            if exposure_outcome == "demoted" and exposure_stage == 0:
                candidate_cooldown_until = self._step_id + max(
                    1,
                    int(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_failure_cooldown_steps",
                            32,
                        )
                    ),
                )
                self._cooldown_until[
                    (bucket.as_tuple(), pending.selected_m)
                ] = candidate_cooldown_until
        self._write_trace({
            "type": "adaptive_ubatch_contextual_observation",
            "step_id": self._step_id,
            "bucket": bucket.as_tuple(),
            "selected_m": pending.selected_m,
            "prior_ms": pending.prior_ms,
            "actual_ms": pending.actual_ms,
            "baseline_ms": pending.baseline_ms,
            "relative_ratio": (
                pending.actual_ms / max(pending.baseline_ms, 1e-9)
            ),
            "model_target_kind": (
                "safe_absolute_scale"
                if pending.selected_m == safe_m
                else "candidate_relative_to_safe"
            ),
            "model_log_target": target,
            "model_raw_log_target": raw_target,
            "transition_sample": pending.transition_sample,
            "affects_steady_state_model": not pending.transition_sample,
            "model_innovation_clipped": not math.isclose(
                target,
                raw_target,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ),
            "regret_ms": regret_ms,
            "raw_regret_ms": raw_regret_ms,
            "raw_regret_pct": raw_regret_pct,
            "rolling_regret_pct": regret_state.regret_pct(),
            "candidate_regret_pct": (
                self._arm_regret_state(
                    bucket,
                    pending.selected_m,
                ).regret_pct()
                if pending.selected_m != safe_m
                else 0.0
            ),
            "safe_anchor_scale": regime_state.safe_anchor_scale,
            "safe_anchor_step": regime_state.safe_anchor_step,
            "candidate_cooldown_until": candidate_cooldown_until,
            "queue_growth": queue_growth,
            "waiting_growth": waiting_growth,
            "queue_regressed": queue_regressed,
            "safe_queue_growth_mean": safe_queue_growth_mean,
            "safe_queue_positive_ratio": safe_queue_positive_ratio,
            "contextual_count": state.count,
            "exposure_stage": exposure_stage,
            "exposure_outcome": exposure_outcome,
        })

    def _apply_contextual_safety(
        self,
        *,
        proposed: CandidateScore,
        safe_score: CandidateScore,
        candidate_scores: list[CandidateScore],
        bucket: WorkloadBucket,
        context: tuple[float, ...],
        regime_warmup: bool,
    ) -> tuple[
        CandidateScore,
        str,
        float | None,
        float,
        float,
        dict[int, dict[str, Any]],
    ]:
        context_array = np.asarray(context, dtype=np.float64)
        regime_state = self._regime_state(bucket)
        safe_cost, _, safe_count = self._predict_safe_contextual_cost(
            score=safe_score, context=context_array
        )
        contextual_safe_cost = safe_cost
        anchor_valid = self._safe_anchor_is_valid(
            regime=regime_state,
            context=context,
        )
        if anchor_valid and regime_state.safe_anchor_scale is not None:
            # Transfer the immediately measured M=1 scale to the current
            # analytical safe cost. This is a local counterfactual anchor,
            # not a workload-specific constant.
            anchored_safe_cost = max(
                1e-6,
                safe_score.prior_cost_ms * regime_state.safe_anchor_scale,
            )
            correction_ratio = max(
                0.0,
                float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_max_correction_ratio",
                        0.3,
                    )
                ),
            )
            safe_cost = max(
                contextual_safe_cost / (1.0 + correction_ratio),
                min(
                    contextual_safe_cost * (1.0 + correction_ratio),
                    anchored_safe_cost,
                ),
            )
        regret_state = self._regret_state(bucket)
        regret_window_steps = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_regret_window_steps",
                    64,
                )
            ),
        )
        # Record one safe-policy exposure for every decision, not only the
        # sparsely sampled timing-feedback steps. Observed candidate loss is
        # charged to this entry when the pending outcome is finalized.
        regret_state.add(
            regret_ms=0.0,
            baseline_ms=safe_cost,
            window_steps=regret_window_steps,
        )
        regret_pct = regret_state.regret_pct()
        risk_kappa = max(
            0.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_risk_kappa",
                    1.0,
                )
            ),
        )
        min_observations = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_context_min_observations",
                    3,
                )
            ),
        )
        min_gain = float(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_min_gain_pct",
                5.0,
            )
        )
        evaluations: dict[int, dict[str, Any]] = {}
        proven: list[tuple[float, CandidateScore]] = []
        exploratory: list[tuple[int, float, CandidateScore]] = []
        max_exploration_regret = max(
            0.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_max_exploration_regret_pct",
                    5.0,
                )
            ),
        )
        budget_pct = max(
            0.0,
            float(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_regret_budget_pct",
                    2.0,
                )
            ),
        )
        regret_state = self._regret_state(bucket)
        queue_safety_enabled = bool(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_queue_safety_enabled",
                True,
            )
        )
        stable_steps = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exploration_stable_steps",
                    8,
                )
            ),
        )
        queue_min_samples = max(3, stable_steps)
        safe_queue_growth_mean = regret_state.safe_queue_growth_mean(
            min_samples=queue_min_samples,
        )
        safe_queue_positive_ratio = regret_state.safe_queue_positive_ratio(
            min_samples=queue_min_samples,
        )
        confirmations = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_switch_confirmations",
                    2,
                )
            ),
        )
        # A persistent sequence of positive deltas matters more than their
        # magnitude because arrivals make individual queue jumps bursty. The
        # limit is derived from the contextual evidence window.
        queue_positive_ratio_limit = confirmations / max(
            1.0,
            min_observations + confirmations,
        )
        queue_pressure = (
            min(
                1.0,
                safe_queue_positive_ratio / queue_positive_ratio_limit,
            )
            if queue_safety_enabled
            else 0.0
        )
        # Under a rising safe-M queue, only a strongly proven arm may be
        # exploited. The extra margin is derived from the configured regret
        # bound and confirmation count; it is not tied to a model or QPS.
        required_gain_pct = min_gain + (
            queue_pressure
            * max_exploration_regret
            * math.sqrt(confirmations)
        )
        for score in candidate_scores:
            if score.m == safe_score.m or score.rejected:
                continue
            arm_regret_state = self._arm_regret_state(bucket, score.m)
            arm_regret_state.add(
                regret_ms=0.0,
                baseline_ms=safe_cost,
                window_steps=regret_window_steps,
            )
            arm_regret_pct = arm_regret_state.regret_pct()
            ratio, uncertainty, count = self._predict_relative_candidate(
                score=score,
                safe_score=safe_score,
                context=context_array,
            )
            relative_context = _relative_context_vector(
                context_array,
                candidate_prior_ms=score.prior_cost_ms,
                safe_prior_ms=safe_score.prior_cost_ms,
            )
            contextual_state = self._contextual_state(score.m)
            (
                offline_log_ratio,
                online_residual_log_ratio,
                _,
            ) = contextual_state.predict_components(relative_context)
            offline_point_gain_pct = (
                1.0 - math.exp(offline_log_ratio)
            ) * 100.0
            upper_ratio = ratio * math.exp(risk_kappa * uncertainty)
            lower_ratio = ratio * math.exp(-risk_kappa * uncertainty)
            gain_lcb_pct = (1.0 - upper_ratio) * 100.0
            gain_ucb_pct = (1.0 - lower_ratio) * 100.0
            point_gain_pct = (1.0 - ratio) * 100.0
            exploration_attempts = regime_state.exploration_attempts.get(
                score.m,
                0,
            )
            exposure_state = self._exposure_state(bucket, score.m)
            exposure_ratios = self._exposure_ratios()
            exposure_ratio = (
                exposure_ratios[exposure_state.stage - 1]
                if exposure_state.stage > 0
                else 0.0
            )
            regime_validated = (
                regime_state.generation == 0
                or exploration_attempts > 0
            )
            evaluations[score.m] = {
                "contextual_model_source": (
                    "offline_robust_prior"
                    if contextual_state.has_offline_prior and count == 0
                    else "offline_plus_paired_residual"
                ),
                "contextual_offline_point_gain_pct": (
                    offline_point_gain_pct
                ),
                "contextual_online_residual_log_ratio": (
                    online_residual_log_ratio
                ),
                "contextual_online_gain_adjustment_pct": (
                    point_gain_pct - offline_point_gain_pct
                ),
                "contextual_analytical_log_ratio": math.log(
                    max(score.prior_cost_ms, 1e-9)
                    / max(safe_score.prior_cost_ms, 1e-9)
                ),
                "contextual_relative_ratio": ratio,
                "contextual_log_uncertainty": uncertainty,
                "contextual_gain_lcb_pct": gain_lcb_pct,
                "contextual_gain_ucb_pct": gain_ucb_pct,
                "contextual_point_gain_pct": point_gain_pct,
                "contextual_count": count,
                "contextual_required_gain_pct": required_gain_pct,
                "contextual_queue_pressure": queue_pressure,
                "contextual_safe_queue_growth_mean": (
                    safe_queue_growth_mean
                ),
                "contextual_safe_queue_positive_ratio": (
                    safe_queue_positive_ratio
                ),
                "contextual_regime_generation": regime_state.generation,
                "contextual_regime_exploration_attempts": (
                    exploration_attempts
                ),
                "contextual_candidate_regret_pct": arm_regret_pct,
                "contextual_baseline_source": (
                    "paired_safe_anchor"
                    if anchor_valid
                    else "contextual_safe_model"
                ),
                "contextual_anchor_raw_ms": (
                    anchored_safe_cost if anchor_valid else None
                ),
                "contextual_exposure_stage": exposure_state.stage,
                "contextual_exposure_ratio": exposure_ratio,
                "contextual_exposure_validated": exposure_state.validated,
                "contextual_exposure_validation_samples": len(
                    exposure_state.validation_ratios
                ),
            }
            if (
                count >= min_observations
                and regime_validated
                and gain_lcb_pct >= required_gain_pct
                and arm_regret_pct < budget_pct
            ):
                proven.append((gain_lcb_pct, score))
                continue
            prior_regret_pct = (
                (score.robust_cost_ms - safe_score.robust_cost_ms)
                / max(safe_score.robust_cost_ms, 1e-6)
                * 100.0
            )
            needs_initial_evidence = (
                count < min_observations
                and exploration_attempts < min_observations + 1
                and gain_ucb_pct >= min_gain
            )
            needs_dynamic_validation = (
                regime_state.generation > 0
                and exploration_attempts == 0
            )
            observed_candidate_is_plausible = (
                needs_initial_evidence
                or (
                    needs_dynamic_validation
                    and point_gain_pct >= min_gain
                )
            )
            if (
                observed_candidate_is_plausible
                and prior_regret_pct <= max_exploration_regret
                and arm_regret_pct < budget_pct
            ):
                # Keep the analytical policy as the eligibility prior. Among
                # uncertain arms, collect evidence for the least-observed and
                # lowest-risk M first; M4 remains independently eligible after
                # M2 has received a real (non-compilation) observation.
                exploratory.append((count, -point_gain_pct, score))

        validation = self._validation_state(bucket)
        if (
            validation.phase != "idle"
            and validation.target_bucket != bucket.as_tuple()
        ):
            idle_timeout = max(
                16,
                int(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_exploration_interval_steps",
                        16,
                    )
                )
                * 4,
            )
            if (
                validation.last_target_step_id >= 0
                and self._step_id - validation.last_target_step_id
                > idle_timeout
            ):
                self._cancel_validation("target_bucket_idle_timeout")
                return (
                    safe_score,
                    "contextual_validation_target_timeout",
                    None,
                    safe_cost,
                    regret_pct,
                    evaluations,
                )
            if validation.washout_remaining <= 0:
                self._set_validation_meta(
                    validation,
                    boundary=False,
                    target_step=False,
                )
                reason = "contextual_validation_background"
            else:
                reason = "contextual_validation_background_washout"
            return (
                safe_score,
                reason,
                validation.predicted_gain_lcb_pct,
                safe_cost,
                regret_pct,
                evaluations,
            )

        if regime_warmup:
            validation = self._validation_state(bucket)
            if validation.phase != "idle":
                self._cancel_validation("target_regime_warmup")
            self._clear_probe(regime_state)
            regime_state.pending_probe_m = None
            regime_state.pending_probe_generation = -1
            regime_state.pending_probe_step = -1
            regime_state.post_anchor_pending = False
            return (
                safe_score,
                "contextual_regime_warmup",
                None,
                safe_cost,
                regret_pct,
                evaluations,
            )
        if regime_state.post_anchor_pending:
            regime_state.post_anchor_pending = False
            return (
                safe_score,
                "contextual_probe_post_anchor",
                None,
                safe_cost,
                regret_pct,
                evaluations,
            )
        if regret_pct >= budget_pct:
            validation = self._validation_state(bucket)
            if validation.phase != "idle":
                self._cancel_validation("target_regret_budget")
            if regime_state.probe_m is not None:
                regime_state.post_anchor_pending = True
            self._clear_probe(regime_state)
            regime_state.pending_probe_m = None
            regime_state.pending_probe_generation = -1
            regime_state.pending_probe_step = -1
            return (
                safe_score,
                "contextual_regret_budget",
                None,
                safe_cost,
                regret_pct,
                evaluations,
            )
        # Contextual candidates are validated as contiguous A/B/A serving
        # windows.  This replaces the old percentage-credit exposure, whose
        # isolated M>1 steps forced a PP synchronization on both entry and
        # return to M=1 and therefore measured controller overhead more than
        # the candidate policy.
        validation = self._validation_state(bucket)
        active = self._select_validation_phase(
            state=validation,
            safe_score=safe_score,
            candidate_scores=candidate_scores,
            context=context,
        )
        if active is not None:
            selected, validation_reason = active
            return (
                selected,
                validation_reason,
                evaluations.get(selected.m, {}).get(
                    "contextual_gain_lcb_pct"
                ),
                safe_cost,
                regret_pct,
                evaluations,
            )

        interval = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exploration_interval_steps",
                    16,
                )
            ),
        )
        queue_stable = regret_state.queue_bad_streak == 0
        stable_enough = regime_state.stable_observations >= stable_steps
        candidate_pool = [score for _, score in proven]
        if not candidate_pool and exploratory:
            candidate_pool = [
                min(
                    exploratory,
                    key=lambda item: (item[0], item[2].m, item[1]),
                )[2]
            ]
        if (
            candidate_pool
            and queue_stable
            and stable_enough
            and safe_count >= min_observations
            and self._step_id - self._last_explore_step >= interval
        ):
            selected_candidate = max(
                candidate_pool,
                key=lambda score: evaluations.get(score.m, {}).get(
                    "contextual_gain_lcb_pct", -math.inf
                ),
            )
            self._last_explore_step = self._step_id
            validation.candidate_m = selected_candidate.m
            validation.target_bucket = bucket.as_tuple()
            validation.generation = regime_state.generation
            validation.stage = 0
            validation.phase = "safe_before"
            validation.phase_step = 0
            self._next_validation_window_id += 1
            validation.window_id = self._next_validation_window_id
            validation.safe_before_rate = None
            validation.candidate_rate = None
            validation.predicted_point_gain_pct = evaluations.get(
                selected_candidate.m, {}
            ).get("contextual_point_gain_pct")
            validation.predicted_gain_lcb_pct = evaluations.get(
                selected_candidate.m, {}
            ).get("contextual_gain_lcb_pct")
            self._next_validation_experiment_id += 1
            validation.experiment_id = self._next_validation_experiment_id
            validation.last_target_step_id = self._step_id
            active = self._select_validation_phase(
                state=validation,
                safe_score=safe_score,
                candidate_scores=candidate_scores,
                context=context,
            )
            assert active is not None
            selected, validation_reason = active
            return (
                selected,
                validation_reason,
                evaluations.get(selected_candidate.m, {}).get(
                    "contextual_gain_lcb_pct"
                ),
                safe_cost,
                regret_pct,
                evaluations,
            )
        return (
            safe_score,
            "contextual_validation_wait",
            max(
                (
                    evaluation["contextual_gain_lcb_pct"]
                    for evaluation in evaluations.values()
                ),
                default=None,
            ),
            safe_cost,
            regret_pct,
            evaluations,
        )
        pending_probe = next(
            (
                score
                for score in candidate_scores
                if score.m == regime_state.pending_probe_m
                and not score.rejected
            ),
            None,
        )
        if (
            pending_probe is not None
            and regime_state.pending_probe_generation
            == regime_state.generation
        ):
            anchor_ready = (
                anchor_valid
                and regime_state.safe_anchor_step
                >= regime_state.pending_probe_step
            )
            if not anchor_ready:
                return (
                    safe_score,
                    "contextual_probe_pre_anchor",
                    evaluations.get(pending_probe.m, {}).get(
                        "contextual_gain_lcb_pct"
                    ),
                    safe_cost,
                    regret_pct,
                    evaluations,
                )
            regime_state.pending_probe_m = None
            regime_state.pending_probe_generation = -1
            regime_state.pending_probe_step = -1
            regime_state.exploration_attempts[pending_probe.m] = (
                regime_state.exploration_attempts.get(pending_probe.m, 0)
                + 1
            )
            regime_state.probe_m = pending_probe.m
            regime_state.probe_remaining = min_observations
            regime_state.probe_generation = regime_state.generation
            regime_state.probe_context = context
            return (
                pending_probe,
                "contextual_exploration",
                evaluations[pending_probe.m]["contextual_gain_lcb_pct"],
                safe_cost,
                regret_pct,
                evaluations,
            )
        if regime_state.pending_probe_m is not None:
            regime_state.pending_probe_m = None
            regime_state.pending_probe_generation = -1
            regime_state.pending_probe_step = -1
        active_probe = next(
            (
                score
                for score in candidate_scores
                if score.m == regime_state.probe_m and not score.rejected
            ),
            None,
        )
        if (
            active_probe is not None
            and regime_state.probe_remaining > 0
            and regime_state.probe_generation == regime_state.generation
            and regret_state.queue_bad_streak == 0
        ):
            distance_limit = max(
                1e-6,
                float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_context_change_threshold",
                        0.12,
                    )
                ),
            )
            if (
                regime_state.probe_context is not None
                and _context_distance(regime_state.probe_context, context)
                > distance_limit
            ):
                self._clear_probe(regime_state)
                regime_state.post_anchor_pending = True
                return (
                    safe_score,
                    "contextual_probe_context_guard",
                    None,
                    safe_cost,
                    regret_pct,
                    evaluations,
                )
            regime_state.probe_remaining -= 1
            if regime_state.probe_remaining == 0:
                self._clear_probe(regime_state)
                regime_state.post_anchor_pending = True
            evaluation = evaluations.get(active_probe.m, {})
            return (
                active_probe,
                "contextual_probe_lease",
                evaluation.get("contextual_gain_lcb_pct"),
                safe_cost,
                regret_pct,
                evaluations,
            )
        if regime_state.probe_m is not None:
            self._clear_probe(regime_state)
            regime_state.post_anchor_pending = True
        interval = max(
            1,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_exploration_interval_steps",
                    16,
                )
            ),
        )
        queue_stable = self._regret_state(bucket).queue_bad_streak == 0
        bucket_stable_observations = self._regime_state(
            bucket
        ).stable_observations
        can_explore = (
            bool(exploratory)
            and queue_stable
            and queue_pressure < 1.0
            and safe_count >= min_observations
            and bucket_stable_observations >= stable_steps
            and self._step_id - self._last_explore_step >= interval
        )
        if can_explore:
            self._last_explore_step = self._step_id
            _, _, selected = min(
                exploratory,
                key=lambda item: (item[0], item[2].m, item[1]),
            )
            # Bracket every uncertain candidate lease with a measured safe-M
            # anchor. The following eligible decision starts the contiguous
            # lease only after this safe decision has produced feedback.
            regime_state.pending_probe_m = selected.m
            regime_state.pending_probe_generation = regime_state.generation
            regime_state.pending_probe_step = self._step_id
            return (
                safe_score,
                "contextual_probe_pre_anchor",
                evaluations[selected.m]["contextual_gain_lcb_pct"],
                safe_cost,
                regret_pct,
                evaluations,
            )
        if proven:
            gain_lcb_pct, selected = max(proven, key=lambda item: item[0])
            exposure_state = self._exposure_state(bucket, selected.m)
            exposure_ratios = self._exposure_ratios()
            # A statistically promising arm is not permission to use it on
            # every eligible step. Accumulate deterministic exposure credit so
            # a false positive is initially confined to a small share of live
            # traffic and earns larger shares only through observations.
            if exposure_state.stage <= 0:
                return (
                    safe_score,
                    "contextual_exposure_unvalidated",
                    gain_lcb_pct,
                    safe_cost,
                    regret_pct,
                    evaluations,
                )
            exposure_ratio = exposure_ratios[
                min(exposure_state.stage, len(exposure_ratios)) - 1
            ]
            exposure_state.credit = min(
                1.0,
                exposure_state.credit + exposure_ratio,
            )
            if exposure_state.credit < 1.0 - 1e-12:
                return (
                    safe_score,
                    "contextual_exposure_guard",
                    gain_lcb_pct,
                    safe_cost,
                    regret_pct,
                    evaluations,
                )
            exposure_state.credit -= 1.0
            validation_target = max(
                1,
                int(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_exposure_validation_observations",
                        3,
                    )
                ),
            )
            validating = (
                not exposure_state.validated
                and len(exposure_state.validation_ratios)
                < validation_target
            )
            return (
                selected,
                (
                    "contextual_exposure_validation"
                    if validating
                    else "contextual_bounded_gain"
                ),
                gain_lcb_pct,
                safe_cost,
                regret_pct,
                evaluations,
            )
        best_gain = max(
            (
                evaluation["contextual_gain_lcb_pct"]
                for evaluation in evaluations.values()
            ),
            default=None,
        )
        if queue_pressure >= 1.0 and exploratory:
            reason = "contextual_queue_trend_guard"
        elif (
            queue_pressure > 0.0
            and any(
                evaluation["contextual_gain_lcb_pct"] >= min_gain
                and evaluation["contextual_gain_lcb_pct"]
                < required_gain_pct
                for evaluation in evaluations.values()
            )
        ):
            reason = "contextual_queue_gain_guard"
        else:
            reason = (
                "contextual_base_safe"
                if not evaluations
                or (proposed.m == safe_score.m and not exploratory)
                else "contextual_insufficient_evidence"
            )
        return (
            safe_score,
            reason,
            best_gain,
            safe_cost,
            regret_pct,
            evaluations,
        )

    def _score_candidates(
        self,
        *,
        features: AdaptiveUBatchFeatures,
        bucket: WorkloadBucket,
        candidates: list[int],
        current_m: int,
        safe_m: int,
    ) -> list[CandidateScore]:
        mode = self._mode()
        risk_enabled = mode in {"calibrated_risk_aware", "contextual_safe"}
        calibration_enabled = mode in {
            "calibrated",
            "calibrated_risk_aware",
            "contextual_safe",
        }
        risk_kappa = float(
            getattr(self.parallel_config, "adaptive_ubatch_risk_kappa", 1.0)
        )
        max_uncertainty_ratio = float(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_max_uncertainty_ratio",
                0.15,
            )
        )
        scores: list[CandidateScore] = []
        for m in candidates:
            prior = _analytical_cost_ms(m, features)
            if not math.isfinite(prior) or prior <= 0:
                scores.append(CandidateScore(
                    m=m,
                    prior_cost_ms=prior,
                    correction_ms=0.0,
                    calibrated_cost_ms=max(prior, 1e-6),
                    uncertainty_ms=float("inf"),
                    robust_cost_ms=float("inf"),
                    count=0,
                    rejected=True,
                    rejection_reason="invalid_prior_cost",
                ))
                continue
            state = self._lookup_state(bucket, m)
            scale = (
                self._calibration_scale(state)
                if calibration_enabled
                else 1.0
            )
            calibrated = max(1e-6, prior * scale)
            correction = calibrated - prior
            uncertainty = (
                self._uncertainty_ms(calibrated, state)
                if calibration_enabled
                else 0.0
            )
            robust = calibrated + (risk_kappa * uncertainty if risk_enabled else 0.0)
            rejected = False
            rejection_reason = None
            if risk_enabled:
                cooldown_key = (bucket.as_tuple(), m)
                if (
                    m != safe_m
                    and self._cooldown_until.get(cooldown_key, -1) > self._step_id
                ):
                    rejected = True
                    rejection_reason = "cooldown"
                elif mode == "calibrated_risk_aware" and (
                    state.count >= self._min_observations()
                    and uncertainty / max(calibrated, 1e-6) > max_uncertainty_ratio
                    and m not in {current_m, safe_m}
                ):
                    rejected = True
                    rejection_reason = "uncertainty_too_high"
            scores.append(CandidateScore(
                m=m,
                prior_cost_ms=prior,
                correction_ms=correction,
                calibrated_cost_ms=calibrated,
                uncertainty_ms=uncertainty,
                robust_cost_ms=robust,
                count=state.count,
                last_update_step=state.last_update_step,
                rejected=rejected,
                rejection_reason=rejection_reason,
            ))
        return scores

    def _select_exploration(
        self,
        *,
        scores: list[CandidateScore],
        current_score: CandidateScore,
        bad_streak: int,
    ) -> CandidateScore | None:
        if self._mode() != "calibrated_risk_aware":
            return None
        if not bool(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_enable_exploration",
                float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_explore_pct",
                        0.0,
                    )
                ) > 0,
            )
        ):
            return None
        interval = int(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_exploration_interval_steps",
                64,
            )
            or 64
        )
        stable_steps = int(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_exploration_stable_steps",
                8,
            )
            or 8
        )
        if self._step_id - self._last_explore_step < interval:
            return None
        if self._stable_bucket_steps < stable_steps or bad_streak > 0:
            return None
        current_m = current_score.m
        adjacent = [
            s for s in scores
            if not s.rejected and s.m != current_m and abs(s.m - current_m) <= 2
        ]
        if not adjacent:
            return None
        # Dynamic request rates can invalidate a previously learned ordering
        # without changing model/phase/token bucket. Re-sample the stalest
        # candidate at a low bounded frequency even when its old prediction is
        # outside the normal exploration regret band.
        stale_after = max(interval * 4, 64)
        stale = [
            score
            for score in adjacent
            if (
                score.count > 0
                and self._step_id - score.last_update_step >= stale_after
            )
        ]
        if stale:
            return min(
                stale,
                key=lambda score: (
                    score.last_update_step,
                    score.robust_cost_ms,
                ),
            )
        max_regret = float(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_max_exploration_regret_pct",
                5.0,
            )
        )
        safe = [
            s for s in adjacent
            if (
                (s.robust_cost_ms - current_score.robust_cost_ms)
                / max(current_score.robust_cost_ms, 1e-6)
                * 100.0
            ) <= max_regret
        ]
        if not safe:
            return None
        return min(safe, key=lambda s: (s.count, s.robust_cost_ms))

    def _write_trace(self, record: dict[str, Any]) -> None:
        if not self._trace_path:
            return
        if self._trace_file is None:
            directory = os.path.dirname(self._trace_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            # Preserve one JSON object per line, but avoid an open/close pair
            # for every decision and observation.
            self._trace_file = open(
                self._trace_path,
                "a",
                encoding="utf-8",
                buffering=256 * 1024,
            )
        self._trace_file.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._trace_records_since_flush += 1
        if (
            self._trace_records_since_flush >= 128
            or str(record.get("type", "")).endswith("failure")
        ):
            self._trace_file.flush()
            self._trace_records_since_flush = 0

    def close_trace(self) -> None:
        trace_file = self._trace_file
        self._trace_file = None
        self._trace_records_since_flush = 0
        if trace_file is not None and not trace_file.closed:
            trace_file.close()

    def __del__(self) -> None:
        try:
            self.close_trace()
        except Exception:
            # Interpreter shutdown may already have torn down file objects.
            pass

    def commit_effective_m(
        self,
        decision: AdaptiveUBatchDecision,
        *,
        effective_m: int,
        reason: str,
        fallback: bool | None = None,
    ) -> AdaptiveUBatchDecision:
        """Synchronize controller state with the M that will actually execute."""
        with self._lock:
            effective_m = max(1, int(effective_m))
            bucket = WorkloadBucket.from_key(decision.bucket_key)
            bucket_state = self._decision_state(bucket)
            previous_m = (
                decision.previous_m
                if decision.previous_m is not None
                else bucket_state.current_m
            )
            switched = effective_m != previous_m
            self._current_m = effective_m
            bucket_state.current_m = effective_m
            bucket_state.pending_m = None
            bucket_state.pending_wins = 0
            if switched:
                self._last_change_step = self._step_id
                bucket_state.last_change_step = self._step_id
            return AdaptiveUBatchDecision(
                num_ubatches=effective_m,
                predicted_gain_pct=decision.predicted_gain_pct,
                reason=f"{decision.reason}; {reason}",
                bucket_key=decision.bucket_key,
                total_tokens=decision.total_tokens,
                online=decision.online,
                num_reqs=decision.num_reqs,
                predicted_cost_ms=decision.predicted_cost_ms,
                robust_cost_ms=decision.robust_cost_ms,
                previous_m=previous_m,
                switched=switched,
                fallback=decision.fallback if fallback is None else fallback,
                decision_overhead_us=decision.decision_overhead_us,
                candidate_scores=decision.candidate_scores,
                queue_depth=decision.queue_depth,
                waiting_reqs=decision.waiting_reqs,
                context_vector=decision.context_vector,
                contextual_baseline_ms=decision.contextual_baseline_ms,
                contextual_gain_lcb_pct=decision.contextual_gain_lcb_pct,
            contextual_regret_pct=decision.contextual_regret_pct,
            service_output_tokens=decision.service_output_tokens,
            service_completed_reqs=decision.service_completed_reqs,
            max_waiting_age_ms=decision.max_waiting_age_ms,
            validation_window_id=decision.validation_window_id,
            validation_phase=decision.validation_phase,
            validation_boundary=decision.validation_boundary,
            validation_target_bucket=decision.validation_target_bucket,
            validation_target_step=decision.validation_target_step,
            validation_stage=decision.validation_stage,
            validation_experiment_id=decision.validation_experiment_id,
        )

    def observe_rejection(
        self,
        decision: AdaptiveUBatchDecision | None,
        *,
        reason: str,
    ) -> None:
        """Record a candidate rejection before execution without bad streak."""
        with self._lock:
            if decision is None:
                return
            bucket = WorkloadBucket.from_key(decision.bucket_key)
            m = max(1, int(decision.num_ubatches))
            safe_m = max(
                1,
                int(getattr(self.parallel_config, "adaptive_ubatch_safe_m", 1)),
            )
            affects_cooldown = m != safe_m
            if affects_cooldown:
                self._cooldown_until[(bucket.as_tuple(), m)] = (
                    self._step_id
                    + int(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_cooldown_steps",
                            8,
                        )
                    )
                )
            if (
                decision.validation_phase is not None
                and decision.validation_target_step
                and m != safe_m
            ):
                self._cancel_validation("candidate_execution_rejected")
            event_type = (
                "adaptive_ubatch_unsupported_fallback"
                if reason.startswith("unsupported:")
                else "adaptive_ubatch_rejection"
            )
            self._write_trace({
                "type": event_type,
                "step_id": self._step_id,
                "bucket": bucket.as_tuple(),
                "rejected_m": m,
                "reason": reason,
                "affects_cooldown": affects_cooldown,
            })

    def select(
        self,
        num_scheduled_tokens: np.ndarray | list[int],
        *,
        queue_depth: int | None = None,
        waiting_reqs: int | None = None,
        service_output_tokens: int = 0,
        service_completed_reqs: int = 0,
        max_waiting_age_ms: float = 0.0,
    ) -> AdaptiveUBatchDecision:
        start_ns = time.perf_counter_ns()
        with self._lock:
            self._step_id += 1
            self._last_validation_meta = (-1, None, False)
            self._last_validation_target_bucket = None
            self._last_validation_target_step = False
            self._last_validation_stage = -1
            self._last_validation_experiment_id = -1
            features = extract_adaptive_ubatch_features(
                model_config=self.model_config,
                num_scheduled_tokens=num_scheduled_tokens,
            )
            bucket = WorkloadBucket.from_key(features.bucket_key)
            contextual_mode = self._mode() == "contextual_safe"
            context = _context_vector(
                features,
                queue_depth=queue_depth,
                waiting_reqs=waiting_reqs,
            )
            regime_warmup = False
            context_distance = 0.0
            if contextual_mode:
                self._finalize_queue_outcome(
                    queue_depth=queue_depth,
                    waiting_reqs=waiting_reqs,
                )
                self._finalize_contextual_outcome(
                    queue_depth=queue_depth,
                    waiting_reqs=waiting_reqs,
                )
                regime_warmup, context_distance = self._update_context_regime(
                    bucket=bucket,
                    context=context,
                )
            if self._last_bucket == bucket:
                self._stable_bucket_steps += 1
            else:
                self._stable_bucket_steps = 1
                self._last_bucket = bucket
            bucket_state = self._decision_state(bucket)

            if features.total_tokens <= 1 or features.num_reqs <= 1:
                validation = self._active_validation
                if (
                    contextual_mode
                    and validation.phase != "idle"
                    and validation.washout_remaining <= 0
                ):
                    idle_timeout = max(
                        16,
                        int(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_exploration_interval_steps",
                                16,
                            )
                        )
                        * 4,
                    )
                    if (
                        validation.last_target_step_id >= 0
                        and self._step_id - validation.last_target_step_id
                        > idle_timeout
                    ):
                        self._cancel_validation(
                            "target_bucket_idle_timeout"
                        )
                    else:
                        self._set_validation_meta(
                            validation,
                            boundary=False,
                            target_step=False,
                        )
                prior_ms = _analytical_cost_ms(1, features)
                calibration = self._lookup_state(bucket, 1)
                calibrated_ms = prior_ms * self._calibration_scale(calibration)
                decision = AdaptiveUBatchDecision(
                    num_ubatches=1,
                    predicted_gain_pct=0.0,
                    reason="too_few_tokens_or_requests",
                    bucket_key=features.bucket_key,
                    total_tokens=features.total_tokens,
                    online=self._mode() != "analytical_only",
                    num_reqs=features.num_reqs,
                    predicted_cost_ms=calibrated_ms,
                    robust_cost_ms=calibrated_ms,
                    previous_m=bucket_state.current_m,
                    switched=False,
                    fallback=True,
                    decision_overhead_us=(
                        time.perf_counter_ns() - start_ns
                    )
                    / 1000.0,
                    candidate_scores=(
                        CandidateScore(
                            m=1,
                            prior_cost_ms=prior_ms,
                            correction_ms=calibrated_ms - prior_ms,
                            calibrated_cost_ms=calibrated_ms,
                            uncertainty_ms=0.0,
                            robust_cost_ms=calibrated_ms,
                            count=calibration.count,
                        ).to_payload(),
                    ),
                    queue_depth=queue_depth,
                    waiting_reqs=waiting_reqs,
                    context_vector=context if contextual_mode else (),
                    contextual_baseline_ms=(
                        calibrated_ms if contextual_mode else None
                    ),
                    contextual_regret_pct=(
                        self._regret_state(bucket).regret_pct()
                        if contextual_mode
                        else 0.0
                    ),
                    service_output_tokens=max(
                        0, int(service_output_tokens)
                    ),
                    service_completed_reqs=max(
                        0, int(service_completed_reqs)
                    ),
                    max_waiting_age_ms=max(
                        0.0, float(max_waiting_age_ms)
                    ),
                    validation_window_id=self._last_validation_meta[0],
                    validation_phase=self._last_validation_meta[1],
                    validation_boundary=self._last_validation_meta[2],
                    validation_target_bucket=(
                        self._last_validation_target_bucket
                    ),
                    validation_target_step=False,
                    validation_stage=self._last_validation_stage,
                    validation_experiment_id=(
                        self._last_validation_experiment_id
                    ),
                )
                self._remember_queue_outcome(decision)
                self._write_trace({
                    "type": "adaptive_ubatch_ineligible",
                    "step_id": self._step_id,
                    "bucket": bucket.as_tuple(),
                    "reason": decision.reason,
                    "selected_m": 1,
                    "validation_window_id": decision.validation_window_id,
                    "validation_phase": decision.validation_phase,
                    "validation_target_bucket": (
                        decision.validation_target_bucket
                    ),
                    "validation_target_step": False,
                    "validation_stage": decision.validation_stage,
                    "affects_cooldown": False,
                    "features": {
                        "total_tokens": features.total_tokens,
                        "num_reqs": features.num_reqs,
                        "max_query_len": features.max_query_len,
                        "avg_tokens_per_req": (
                            features.avg_tokens_per_req
                        ),
                        "token_imbalance": features.token_imbalance,
                        "smallest_request_ratio": (
                            features.smallest_request_ratio
                        ),
                        "prefill_reqs": features.prefill_reqs,
                        "decode_reqs": features.decode_reqs,
                        "prefill_ratio": features.prefill_ratio,
                        "model_billions": features.model_billions,
                        "queue_depth": queue_depth,
                        "waiting_reqs": waiting_reqs,
                    },
                })
                return decision

            if self._mode() == "analytical_only":
                decision = _with_feature_counts(select_adaptive_ubatch_count(
                    parallel_config=self.parallel_config,
                    model_config=self.model_config,
                    num_scheduled_tokens=num_scheduled_tokens,
                ), features)
                return AdaptiveUBatchDecision(
                    num_ubatches=decision.num_ubatches,
                    predicted_gain_pct=decision.predicted_gain_pct,
                    reason=f"analytical_only; prior={decision.reason}",
                    bucket_key=decision.bucket_key,
                    total_tokens=decision.total_tokens,
                    online=False,
                    num_reqs=decision.num_reqs,
                    predicted_cost_ms=decision.predicted_cost_ms,
                    robust_cost_ms=decision.robust_cost_ms,
                    previous_m=bucket_state.current_m,
                    switched=decision.num_ubatches != bucket_state.current_m,
                    decision_overhead_us=(time.perf_counter_ns() - start_ns) / 1000.0,
                )

            candidates = _candidate_ms(
                parallel_config=self.parallel_config,
                features=features,
            )
            if not candidates:
                candidates = [1]
            safe_m = max(
                1,
                min(
                    int(getattr(self.parallel_config, "adaptive_ubatch_safe_m", 1)),
                    max(candidates),
                ),
            )
            previous_m = (
                bucket_state.current_m
                if bucket_state.current_m in candidates
                else safe_m
            )
            scores = self._score_candidates(
                features=features,
                bucket=bucket,
                candidates=candidates,
                current_m=previous_m,
                safe_m=safe_m,
            )
            valid_scores = [s for s in scores if not s.rejected]
            current_score = next(
                (s for s in scores if s.m == previous_m and not s.rejected),
                None,
            )
            if current_score is None:
                current_score = next((s for s in scores if s.m == safe_m), scores[0])
            safe_score = next(
                (score for score in scores if score.m == safe_m),
                None,
            )

            reason = "robust_cost_improvement"
            fallback = False
            if not valid_scores:
                selected = current_score
                reason = "all_candidates_rejected"
                fallback = True
            else:
                safe_score = next(
                    (
                        score
                        for score in valid_scores
                        if score.m == safe_m
                    ),
                    None,
                )
                selected = None
                calibration_target = max(
                    0,
                    int(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_candidate_calibration_observations",
                            3,
                        )
                    ),
                )
                under_observed = [
                    score
                    for score in valid_scores
                    if score.count < calibration_target
                ]
                if (
                    calibration_target > 0
                    and len(valid_scores) > 1
                    and under_observed
                ):
                    # Fully calibrate the current candidate before moving to
                    # the next M. This bounds switching during warmup while
                    # guaranteeing equal coverage for M=1/2/4.
                    selected = next(
                        (
                            score
                            for score in under_observed
                            if score.m == previous_m
                        ),
                        min(under_observed, key=lambda score: score.m),
                    )
                    reason = "candidate_calibration"
                    bucket_state.pending_m = None
                    bucket_state.pending_wins = 0
                else:
                    exploration = self._select_exploration(
                        scores=valid_scores,
                        current_score=current_score,
                        bad_streak=bucket_state.bad_streak,
                    )
                    if exploration is not None:
                        selected = exploration
                        reason = "controlled_exploration"
                        self._last_explore_step = self._step_id
                    else:
                        best = min(valid_scores, key=lambda s: s.robust_cost_ms)
                        safe_score = next(
                            (
                                score
                                for score in valid_scores
                                if score.m == safe_m
                            ),
                            None,
                        )
                        min_gain = float(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_switch_threshold_pct",
                                getattr(
                                    self.parallel_config,
                                    "adaptive_ubatch_min_gain_pct",
                                    5.0,
                                ),
                            ),
                        )
                        safe_gain_pct = None
                        if (
                            safe_score is not None
                            and best.m != safe_m
                        ):
                            safe_gain_pct = (
                                (
                                    safe_score.robust_cost_ms
                                    - best.robust_cost_ms
                                )
                                / max(safe_score.robust_cost_ms, 1e-6)
                                * 100.0
                            )
                        if (
                            safe_score is not None
                            and safe_gain_pct is not None
                            and safe_gain_pct < min_gain
                        ):
                            # The safe candidate is the explicit online
                            # reference. A non-safe M must beat it by the full
                            # switch margin; a marginal win is not enough to
                            # pay adaptive split/communication overhead.
                            selected = safe_score
                            reason = "safe_m_gain_guard"
                            bucket_state.pending_m = None
                            bucket_state.pending_wins = 0
                        else:
                            selected = best
                        min_hold = int(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_min_hold_steps",
                                4,
                            )
                        )
                        if reason == "safe_m_gain_guard":
                            # Safety fallback bypasses hold/confirmation so a
                            # marginal non-safe choice cannot linger.
                            pass
                        elif best.m == previous_m:
                            selected = best
                            reason = "keep_current_best"
                            bucket_state.pending_m = None
                            bucket_state.pending_wins = 0
                        elif self._step_id - bucket_state.last_change_step < min_hold:
                            selected = current_score
                            reason = "minimum_hold"
                        else:
                            gain_pct = (
                                (
                                    current_score.robust_cost_ms
                                    - best.robust_cost_ms
                                )
                                / max(current_score.robust_cost_ms, 1e-6)
                                * 100.0
                            )
                            if gain_pct < min_gain:
                                selected = current_score
                                reason = "gain_below_switch_threshold"
                                bucket_state.pending_m = None
                                bucket_state.pending_wins = 0
                            else:
                                confirmations = max(
                                    1,
                                    int(
                                        getattr(
                                            self.parallel_config,
                                            "adaptive_ubatch_switch_confirmations",
                                            2,
                                        )
                                    ),
                                )
                                if bucket_state.pending_m == best.m:
                                    bucket_state.pending_wins += 1
                                else:
                                    bucket_state.pending_m = best.m
                                    bucket_state.pending_wins = 1
                                if bucket_state.pending_wins < confirmations:
                                    selected = current_score
                                    reason = "switch_confirmation"
                                else:
                                    selected = best
                                    bucket_state.pending_m = None
                                    bucket_state.pending_wins = 0

            contextual_gain_lcb_pct = None
            contextual_baseline_ms = None
            contextual_regret_pct = 0.0
            contextual_evaluations: dict[int, dict[str, Any]] = {}
            if contextual_mode and safe_score is not None:
                (
                    selected,
                    reason,
                    contextual_gain_lcb_pct,
                    contextual_baseline_ms,
                    contextual_regret_pct,
                    contextual_evaluations,
                ) = self._apply_contextual_safety(
                    proposed=selected,
                    safe_score=safe_score,
                    candidate_scores=valid_scores,
                    bucket=bucket,
                    context=context,
                    regime_warmup=regime_warmup,
                )
                if selected.m == safe_m:
                    bucket_state.pending_m = None
                    bucket_state.pending_wins = 0

            predicted_gain_pct = (
                (current_score.robust_cost_ms - selected.robust_cost_ms)
                / max(current_score.robust_cost_ms, 1e-6)
                * 100.0
            )
            selected_m = _cap_candidate(
                selected.m,
                int(getattr(self.parallel_config, "adaptive_ubatch_max_size", 4) or 1),
                features.total_tokens,
            )
            switched = selected_m != previous_m
            if switched:
                self._last_change_step = self._step_id
                self._current_m = selected_m
                bucket_state.last_change_step = self._step_id
                bucket_state.current_m = selected_m
                bucket_state.pending_m = None
                bucket_state.pending_wins = 0
            else:
                self._current_m = previous_m
                bucket_state.current_m = previous_m

            overhead_us = (time.perf_counter_ns() - start_ns) / 1000.0
            validation_window_id, validation_phase, validation_boundary = (
                self._last_validation_meta
            )
            decision = AdaptiveUBatchDecision(
                num_ubatches=selected_m,
                predicted_gain_pct=max(0.0, predicted_gain_pct),
                reason=reason,
                bucket_key=bucket.as_tuple(),
                total_tokens=features.total_tokens,
                online=True,
                num_reqs=features.num_reqs,
                predicted_cost_ms=selected.calibrated_cost_ms,
                robust_cost_ms=selected.robust_cost_ms,
                previous_m=previous_m,
                switched=switched,
                fallback=fallback,
                decision_overhead_us=overhead_us,
                candidate_scores=tuple(
                    {
                        **score.to_payload(),
                        **contextual_evaluations.get(score.m, {}),
                        **(
                            {
                                "contextual_count": self._contextual_state(
                                    score.m
                                ).count,
                                "contextual_role": "safe",
                            }
                            if contextual_mode and score.m == safe_m
                            else {}
                        ),
                    }
                    for score in scores
                ),
                queue_depth=queue_depth,
                waiting_reqs=waiting_reqs,
                context_vector=context if contextual_mode else (),
                contextual_baseline_ms=contextual_baseline_ms,
                contextual_gain_lcb_pct=contextual_gain_lcb_pct,
                contextual_regret_pct=contextual_regret_pct,
                service_output_tokens=max(0, int(service_output_tokens)),
                service_completed_reqs=max(0, int(service_completed_reqs)),
                max_waiting_age_ms=max(0.0, float(max_waiting_age_ms)),
                validation_window_id=validation_window_id,
                validation_phase=validation_phase,
                validation_boundary=validation_boundary,
                validation_target_bucket=(
                    self._last_validation_target_bucket
                ),
                validation_target_step=self._last_validation_target_step,
                validation_stage=self._last_validation_stage,
                validation_experiment_id=(
                    self._last_validation_experiment_id
                ),
            )
            self._remember_queue_outcome(decision)
            self._write_trace({
                "type": "adaptive_ubatch_decision",
                "step_id": self._step_id,
                "bucket": bucket.as_tuple(),
                "features": {
                    "total_tokens": features.total_tokens,
                    "num_reqs": features.num_reqs,
                    "max_query_len": features.max_query_len,
                    "avg_tokens_per_req": features.avg_tokens_per_req,
                    "token_imbalance": features.token_imbalance,
                    "smallest_request_ratio": (
                        features.smallest_request_ratio
                    ),
                    "prefill_reqs": features.prefill_reqs,
                    "decode_reqs": features.decode_reqs,
                    "prefill_ratio": features.prefill_ratio,
                    "model_billions": features.model_billions,
                    "queue_depth": queue_depth,
                    "waiting_reqs": waiting_reqs,
                },
                "previous_m": previous_m,
                "selected_m": selected_m,
                "predicted_gain_pct": decision.predicted_gain_pct,
                "switched": switched,
                "fallback": fallback,
                "reason": reason,
                "validation_window_id": validation_window_id,
                "validation_phase": validation_phase,
                "validation_boundary": validation_boundary,
                "validation_target_bucket": (
                    self._last_validation_target_bucket
                ),
                "validation_target_step": (
                    self._last_validation_target_step
                ),
                "validation_stage": self._last_validation_stage,
                "validation_experiment_id": (
                    self._last_validation_experiment_id
                ),
                "context_distance": context_distance,
                "contextual_stable_observations": (
                    self._regime_state(bucket).stable_observations
                    if contextual_mode
                    else None
                ),
                "contextual_gain_lcb_pct": contextual_gain_lcb_pct,
                "contextual_baseline_ms": contextual_baseline_ms,
                "contextual_regret_pct": contextual_regret_pct,
                "pending_m": bucket_state.pending_m,
                "pending_wins": bucket_state.pending_wins,
                "candidates": list(decision.candidate_scores),
                "decision_overhead_us": overhead_us,
            })
            return decision

    def observe_service_window(
        self,
        decision: AdaptiveUBatchDecision,
        *,
        elapsed_ms: float,
        output_tokens: int,
        completed_reqs: int,
        queue_growth: float = 0.0,
        waiting_age_growth_ms: float = 0.0,
        scheduler_steps: int = 0,
        target_steps: int = 0,
    ) -> None:
        """Commit one boundary-only A/B/A service observation."""
        with self._lock:
            if (
                decision.validation_phase is None
                or not decision.validation_boundary
                or elapsed_ms <= 0.0
            ):
                return
            state = self._active_validation
            if (
                state.window_id != decision.validation_window_id
                or state.phase != decision.validation_phase
                or state.target_bucket != decision.validation_target_bucket
            ):
                return
            bucket = WorkloadBucket.from_key(state.target_bucket)
            min_tokens = max(
                1,
                int(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_validation_min_output_tokens",
                        8,
                    )
                ),
            )
            service_rate = max(0, int(output_tokens)) / max(
                float(elapsed_ms) / 1000.0,
                1e-9,
            )
            target_share = max(0, int(target_steps)) / max(
                1,
                int(scheduler_steps),
            )
            phase = state.phase
            phase_context = self._finish_validation_phase_context(state)
            washout_steps = max(
                0,
                int(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_validation_washout_steps",
                        1,
                    )
                ),
            )
            self._write_trace({
                "type": "adaptive_ubatch_validation_window",
                "step_id": self._step_id,
                "window_id": state.window_id,
                "experiment_id": state.experiment_id,
                "phase": phase,
                "target_bucket": state.target_bucket,
                "candidate_m": state.candidate_m,
                "stage": state.stage,
                "elapsed_ms": float(elapsed_ms),
                "output_tokens": int(output_tokens),
                "completed_reqs": int(completed_reqs),
                "service_rate": service_rate,
                "scheduler_steps": int(scheduler_steps),
                "target_steps": int(target_steps),
                "target_step_share": target_share,
                "background_steps": max(
                    0, int(scheduler_steps) - int(target_steps)
                ),
                "queue_peak_growth": float(queue_growth),
                "waiting_age_peak_growth_ms": float(
                    waiting_age_growth_ms
                ),
                "mean_target_context": phase_context,
                "predicted_point_gain_pct": (
                    state.predicted_point_gain_pct
                ),
                "predicted_gain_lcb_pct": state.predicted_gain_lcb_pct,
            })
            if phase == "safe_before":
                state.safe_before_rate = (
                    service_rate if output_tokens >= min_tokens else None
                )
                state.safe_before_queue_growth = float(queue_growth)
                state.safe_before_target_share = target_share
                state.safe_before_age_growth_ms = float(
                    waiting_age_growth_ms
                )
                state.safe_before_context = phase_context
                min_target_share = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_validation_min_target_share",
                                0.5,
                            )
                        ),
                    ),
                )
                if target_share < min_target_share:
                    self._cancel_validation(
                        "validation_insufficient_target_share"
                    )
                    return
                state.phase = "candidate"
                state.washout_remaining = washout_steps
            elif phase == "candidate":
                state.candidate_rate = (
                    service_rate if output_tokens >= min_tokens else None
                )
                state.candidate_queue_growth = float(queue_growth)
                state.candidate_target_share = target_share
                state.candidate_age_growth_ms = float(
                    waiting_age_growth_ms
                )
                state.candidate_context = phase_context
                if (
                    state.safe_before_target_share is not None
                    and abs(
                        target_share - state.safe_before_target_share
                    )
                    > 2.0 * _VALIDATION_TARGET_SHARE_TOLERANCE
                ):
                    self._cancel_validation(
                        "validation_early_workload_mismatch"
                    )
                    return
                state.phase = "safe_after"
                state.washout_remaining = washout_steps
            elif phase == "safe_after":
                state.safe_after_context = phase_context
                safe_after_rate = (
                    service_rate if output_tokens >= min_tokens else None
                )
                safe_rates = [
                    value
                    for value in (state.safe_before_rate, safe_after_rate)
                    if value is not None and value > 0.0
                ]
                safe_rate = (
                    sum(safe_rates) / len(safe_rates)
                    if len(safe_rates) == 2
                    else None
                )
                gain_pct = (
                    (state.candidate_rate / safe_rate - 1.0) * 100.0
                    if safe_rate and state.candidate_rate
                    else -math.inf
                )
                safe_target_shares = [
                    value
                    for value in (
                        state.safe_before_target_share,
                        target_share,
                    )
                    if value is not None
                ]
                safe_target_share = (
                    sum(safe_target_shares) / len(safe_target_shares)
                    if len(safe_target_shares) == 2
                    else None
                )
                target_share_delta = (
                    abs(state.candidate_target_share - safe_target_share)
                    if state.candidate_target_share is not None
                    and safe_target_share is not None
                    else math.inf
                )
                workload_matched = (
                    target_share_delta <= _VALIDATION_TARGET_SHARE_TOLERANCE
                )
                safe_queue_growth = max(
                    state.safe_before_queue_growth,
                    float(queue_growth),
                )
                safe_age_growth = max(
                    state.safe_before_age_growth_ms,
                    float(waiting_age_growth_ms),
                )
                queue_limit = float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_queue_growth_threshold",
                        2,
                    )
                )
                age_limit = float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_queue_age_growth_threshold_ms",
                        10.0,
                    )
                )
                queue_safe = (
                    state.candidate_queue_growth
                    <= safe_queue_growth + queue_limit
                    and state.candidate_age_growth_ms
                    <= safe_age_growth + age_limit
                )
                min_gain = max(
                    0.0,
                    float(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_validation_gain_pct",
                            2.0,
                        )
                    ),
                )
                candidate_m = int(
                    state.candidate_m or decision.num_ubatches
                )
                evidence = self._validation_evidence_state(
                    bucket,
                    candidate_m,
                    state.stage,
                )
                evidence_context_reset = False
                if (
                    workload_matched
                    and state.candidate_context
                    and evidence.contexts
                    and _context_distance(
                        evidence.contexts[-1],
                        state.candidate_context,
                    )
                    > float(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_context_change_threshold",
                            0.12,
                        )
                    )
                ):
                    evidence.reset(
                        generation=evidence.generation,
                        stage=state.stage,
                    )
                    evidence_context_reset = True
                if workload_matched and math.isfinite(gain_pct):
                    evidence.gains_pct.append(float(gain_pct))
                    evidence.queue_safe.append(bool(queue_safe))
                    evidence.contexts.append(state.candidate_context)
                evidence_count = len(evidence.gains_pct)
                evidence_mean = (
                    sum(evidence.gains_pct) / evidence_count
                    if evidence_count
                    else None
                )
                sorted_gains = sorted(evidence.gains_pct)
                evidence_median = (
                    (
                        sorted_gains[evidence_count // 2]
                        if evidence_count % 2
                        else (
                            sorted_gains[evidence_count // 2 - 1]
                            + sorted_gains[evidence_count // 2]
                        )
                        / 2.0
                    )
                    if evidence_count
                    else None
                )
                evidence_std = 0.0
                if evidence_count > 1 and evidence_mean is not None:
                    evidence_std = math.sqrt(
                        sum(
                            (gain - evidence_mean) ** 2
                            for gain in evidence.gains_pct
                        )
                        / (evidence_count - 1)
                    )
                confidence_kappa = max(
                    0.0,
                    float(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_validation_confidence_kappa",
                            1.0,
                        )
                    ),
                )
                evidence_lcb = (
                    evidence_mean
                    - confidence_kappa
                    * evidence_std
                    / math.sqrt(evidence_count)
                    if evidence_count > 1 and evidence_mean is not None
                    else -math.inf
                )
                required_evidence = max(
                    2,
                    int(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_validation_required_observations",
                            2,
                        )
                    ),
                )
                max_loss_pct = max(
                    0.0,
                    float(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_max_exploration_regret_pct",
                            5.0,
                        )
                    ),
                )
                severe_loss = bool(
                    evidence.gains_pct
                    and min(evidence.gains_pct) < -max_loss_pct
                )
                evidence_ready = evidence_count >= required_evidence
                promoted = bool(
                    workload_matched
                    and evidence_ready
                    and evidence_median is not None
                    and evidence_median >= min_gain
                    and evidence_lcb >= 0.0
                    and all(evidence.queue_safe)
                    and not severe_loss
                )
                validation_continues = bool(
                    workload_matched
                    and queue_safe
                    and not severe_loss
                    and not evidence_ready
                )
                if (
                    workload_matched
                    and safe_rate
                    and state.candidate_rate
                    and state.candidate_context
                    and evidence_ready
                    and evidence_median is not None
                ):
                    # Update once with the aggregate paired label.  Individual
                    # short windows remain diagnostic evidence and cannot
                    # pull the online residual in opposite directions.
                    self._contextual_state(candidate_m).update(
                        context=np.asarray(
                            state.candidate_context,
                            dtype=np.float64,
                        ),
                        target=-math.log1p(
                            max(-99.0, evidence_median) / 100.0
                        ),
                        forgetting_factor=max(
                            1e-3,
                            min(
                                1.0,
                                float(
                                    getattr(
                                        self.parallel_config,
                                        "adaptive_ubatch_context_forgetting_factor",
                                        0.98,
                                    )
                                ),
                            ),
                        ),
                        alpha=self._alpha(),
                        step_id=self._step_id,
                    )
                self._write_trace({
                    "type": "adaptive_ubatch_validation_result",
                    "step_id": self._step_id,
                    "target_bucket": bucket.as_tuple(),
                    "candidate_m": state.candidate_m,
                    "experiment_id": state.experiment_id,
                    "stage": state.stage,
                    "predicted_point_gain_pct": (
                        state.predicted_point_gain_pct
                    ),
                    "predicted_gain_lcb_pct": (
                        state.predicted_gain_lcb_pct
                    ),
                    "candidate_service_rate": state.candidate_rate,
                    "safe_service_rate": safe_rate,
                    "service_gain_pct": gain_pct,
                    # Window timing includes policy washout and full-worker
                    # execution, so this is the measured net serving-rate
                    # label used by the controller.
                    "net_gain_pct": gain_pct,
                    "net_gain_includes_washout": True,
                    "validation_evidence_count": evidence_count,
                    "validation_required_observations": required_evidence,
                    "validation_gain_mean_pct": evidence_mean,
                    "validation_gain_median_pct": evidence_median,
                    "validation_gain_lcb_pct": evidence_lcb,
                    "validation_severe_loss": severe_loss,
                    "validation_evidence_context_reset": (
                        evidence_context_reset
                    ),
                    "safe_target_step_share": safe_target_share,
                    "candidate_target_step_share": (
                        state.candidate_target_share
                    ),
                    "target_step_share_delta": target_share_delta,
                    "workload_matched": workload_matched,
                    "safe_before_context": state.safe_before_context,
                    "candidate_context": state.candidate_context,
                    "safe_after_context": state.safe_after_context,
                    "candidate_queue_peak_growth": (
                        state.candidate_queue_growth
                    ),
                    "candidate_waiting_age_peak_growth_ms": (
                        state.candidate_age_growth_ms
                    ),
                    "queue_safe": queue_safe,
                    "promoted": promoted,
                    "validation_continues": validation_continues,
                })
                if promoted or validation_continues:
                    if promoted:
                        state.stage = min(
                            state.stage + 1,
                            len(self._validation_stage_steps()) - 1,
                        )
                    # The ending A window is the next local counterfactual;
                    # repeat B/A for independent evidence, or use it as the
                    # anchor for the newly promoted lease length.
                    state.safe_before_rate = safe_after_rate
                    state.safe_before_target_share = target_share
                    state.safe_before_context = phase_context
                    state.safe_before_queue_growth = float(queue_growth)
                    state.safe_before_age_growth_ms = float(
                        waiting_age_growth_ms
                    )
                    state.phase = "candidate"
                    state.washout_remaining = washout_steps
                else:
                    if state.candidate_m is not None:
                        self._cooldown_until[
                            (bucket.as_tuple(), state.candidate_m)
                        ] = self._step_id + max(
                            1,
                            int(
                                getattr(
                                    self.parallel_config,
                                    "adaptive_ubatch_failure_cooldown_steps",
                                    32,
                                )
                            ),
                        )
                    self._cancel_validation(
                        (
                            "validation_not_promoted"
                            if workload_matched
                            else "validation_workload_mismatch"
                        )
                    )
                    return
            state.phase_step = 0
            self._next_validation_window_id += 1
            state.window_id = self._next_validation_window_id

    def observe(
        self,
        decision: AdaptiveUBatchDecision,
        *,
        forward_ms: float,
    ) -> None:
        with self._lock:
            if decision is None:
                return
            actual_ms = float(forward_ms)
            predicted_ms = (
                float(decision.predicted_cost_ms)
                if decision.predicted_cost_ms is not None
                else None
            )
            if not _is_finite_positive(actual_ms) or not _is_finite_positive(
                predicted_ms
            ):
                self.observe_failure(decision, reason="invalid_observation")
                return
            bucket = WorkloadBucket.from_key(decision.bucket_key)
            m = max(1, int(decision.num_ubatches))
            bucket_state = self._decision_state(bucket)
            selected_candidate = next(
                (
                    candidate
                    for candidate in decision.candidate_scores
                    if int(candidate.get("m", -1)) == m
                ),
                None,
            )
            safe_m = max(
                1,
                int(getattr(self.parallel_config, "adaptive_ubatch_safe_m", 1)),
            )
            safe_candidate = next(
                (
                    candidate
                    for candidate in decision.candidate_scores
                    if int(candidate.get("m", -1)) == safe_m
                ),
                None,
            )
            prior_ms = (
                float(selected_candidate["prior_ms"])
                if selected_candidate is not None
                and _is_finite_positive(
                    _optional_float(selected_candidate.get("prior_ms"))
                )
                else predicted_ms
            )
            has_alternative = len(decision.candidate_scores) > 1
            cold_key = (bucket.as_tuple(), m)
            decision_reason = decision.reason.split(";", maxsplit=1)[0]
            cold_transition_sample = (
                decision_reason
                in {
                    "candidate_calibration",
                    "contextual_exploration",
                    "contextual_probe_lease",
                }
                and has_alternative
                and cold_key not in self._discarded_cold_samples
            )
            if cold_transition_sample:
                self._discarded_cold_samples.add(cold_key)
                self._write_trace({
                    "type": "adaptive_ubatch_transition_sample",
                    "step_id": self._step_id,
                    "bucket": bucket.as_tuple(),
                    "selected_m": m,
                    "actual_ms": actual_ms,
                    "predicted_ms": predicted_ms,
                    "reason": "first_bucket_m_sample_charged_as_transition",
                    "affects_steady_state_model": False,
                })
            error_ms = actual_ms - predicted_ms
            alpha = self._alpha()
            prior_state_count = self._lookup_state(bucket, m).count
            selected_state = self._state(bucket, m)
            if not cold_transition_sample:
                selected_state.update(
                    error_ms=error_ms,
                    predicted_ms=predicted_ms,
                    prior_ms=prior_ms,
                    actual_ms=actual_ms,
                    step_id=self._step_id,
                    alpha=alpha,
                )
            contextual_mode = self._mode() == "contextual_safe"
            if contextual_mode and decision.context_vector:
                baseline_ms = decision.contextual_baseline_ms
                if _is_finite_positive(baseline_ms):
                    candidate_prior_ms = _optional_float(
                        selected_candidate.get("prior_ms")
                        if selected_candidate is not None
                        else None
                    )
                    safe_prior_ms = _optional_float(
                        safe_candidate.get("prior_ms")
                        if safe_candidate is not None
                        else None
                    )
                    model_context = _relative_context_vector(
                        decision.context_vector,
                        candidate_prior_ms=(
                            candidate_prior_ms
                            if _is_finite_positive(candidate_prior_ms)
                            else prior_ms
                        ),
                        safe_prior_ms=(
                            safe_prior_ms
                            if _is_finite_positive(safe_prior_ms)
                            else prior_ms
                        ),
                    )
                    self._pending_contextual_outcome = PendingContextualOutcome(
                        bucket_key=bucket.as_tuple(),
                        selected_m=m,
                        context=tuple(float(value) for value in model_context),
                        prior_ms=prior_ms,
                        actual_ms=actual_ms,
                        baseline_ms=float(baseline_ms),
                        queue_depth=decision.queue_depth,
                        waiting_reqs=decision.waiting_reqs,
                        transition_sample=cold_transition_sample,
                    )
            degradation_pct = error_ms / max(predicted_ms, 1e-6) * 100.0
            bad_threshold = float(
                getattr(self.parallel_config, "adaptive_ubatch_bad_threshold_pct", 8.0)
            )
            accepted_target = max(
                1,
                int(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_candidate_calibration_observations",
                        3,
                    )
                ),
            )
            selected_state = self._lookup_state(bucket, m)
            safe_state = self._lookup_state(bucket, safe_m)
            relative_safe_regret_pct = None
            bad_vs_safe = False
            if (
                not contextual_mode
                and m != safe_m
                and safe_candidate is not None
                and selected_state.count >= accepted_target
                and safe_state.count >= accepted_target
            ):
                safe_prior_ms = _optional_float(
                    safe_candidate.get("prior_ms")
                )
                if _is_finite_positive(safe_prior_ms):
                    selected_calibrated_ms = (
                        prior_ms * self._calibration_scale(selected_state)
                    )
                    safe_calibrated_ms = (
                        float(safe_prior_ms)
                        * self._calibration_scale(safe_state)
                    )
                    relative_safe_regret_pct = (
                        (
                            selected_calibrated_ms
                            - safe_calibrated_ms
                        )
                        / max(safe_calibrated_ms, 1e-6)
                        * 100.0
                    )
                    regret_threshold = max(
                        3.0,
                        float(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_switch_threshold_pct",
                                5.0,
                            )
                        ),
                    )
                    bad_vs_safe = (
                        relative_safe_regret_pct > regret_threshold
                    )
            enough_observations = prior_state_count >= self._min_observations()
            prediction_bad_execution = (
                has_alternative
                and degradation_pct > bad_threshold
                and (
                    enough_observations
                    or (contextual_mode and m != safe_m)
                )
            )
            # Absolute error against the analytical candidate prediction is
            # not evidence that this arm is worse than M=1. Contextual mode
            # therefore uses the measured safe-arm regret budget and queue
            # response as its emergency brake while retaining this signal for
            # model calibration and diagnostics.
            bad_execution = prediction_bad_execution and not contextual_mode
            bad_non_safe = bad_execution or bad_vs_safe
            if bad_non_safe and m != safe_m:
                bucket_state.bad_streak += 1
                regime_state = self._regime_state(bucket)
                self._clear_probe(regime_state)
                regime_state.pending_probe_m = None
                regime_state.pending_probe_generation = -1
                regime_state.pending_probe_step = -1
                regime_state.post_anchor_pending = True
                self._cooldown_until[(bucket.as_tuple(), m)] = (
                    self._step_id
                    + (
                        max(
                            64,
                            int(
                                getattr(
                                    self.parallel_config,
                                    (
                                        "adaptive_ubatch_"
                                        "failure_cooldown_steps"
                                    ),
                                    32,
                                )
                            )
                            * 4,
                        )
                        if bad_vs_safe or contextual_mode
                        else int(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_cooldown_steps",
                                8,
                            )
                        )
                    )
                )
                self._current_m = safe_m
                bucket_state.current_m = safe_m
                bucket_state.pending_m = None
                bucket_state.pending_wins = 0
            else:
                bucket_state.bad_streak = 0
            self._write_trace({
                "type": "adaptive_ubatch_observation",
                "step_id": self._step_id,
                "bucket": bucket.as_tuple(),
                "selected_m": m,
                "prior_ms": prior_ms,
                "calibration_scale": predicted_ms / max(prior_ms, 1e-6),
                "predicted_ms": predicted_ms,
                "actual_ms": actual_ms,
                "error_ms": error_ms,
                "degradation_pct": degradation_pct,
                "transition_sample": cold_transition_sample,
                "prediction_bad_execution": prediction_bad_execution,
                "bad_execution": bad_execution,
                "relative_safe_regret_pct": relative_safe_regret_pct,
                "bad_vs_safe": bad_vs_safe,
                "has_alternative": has_alternative,
                "enough_observations": enough_observations,
                "contextual_pending": self._pending_contextual_outcome is not None,
                "affects_cooldown": (
                    bad_non_safe and m != safe_m
                ),
            })

    def observe_failure(
        self,
        decision: AdaptiveUBatchDecision | None,
        *,
        reason: str = "runtime_exception",
    ) -> None:
        with self._lock:
            if decision is None:
                return
            bucket = WorkloadBucket.from_key(decision.bucket_key)
            m = max(1, int(decision.num_ubatches))
            bucket_state = self._decision_state(bucket)
            bucket_state.bad_streak += 1
            safe_m = max(
                1,
                int(getattr(self.parallel_config, "adaptive_ubatch_safe_m", 1)),
            )
            affects_cooldown = m != safe_m
            if affects_cooldown:
                self._cooldown_until[(bucket.as_tuple(), m)] = (
                    self._step_id
                    + int(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_failure_cooldown_steps",
                            32,
                        )
                    )
                )
            self._current_m = safe_m
            bucket_state.current_m = safe_m
            bucket_state.pending_m = None
            bucket_state.pending_wins = 0
            if self._active_validation.phase != "idle":
                self._cancel_validation("runtime_failure")
            self._write_trace({
                "type": "adaptive_ubatch_runtime_failure",
                "step_id": self._step_id,
                "bucket": bucket.as_tuple(),
                "selected_m": m,
                "fallback_m": self._current_m,
                "reason": reason,
                "affects_cooldown": affects_cooldown,
            })
