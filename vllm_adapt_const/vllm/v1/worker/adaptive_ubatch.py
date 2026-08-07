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
CONTEXT_DIMENSION = 11


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
    covariance: np.ndarray = field(init=False)
    residual_variance: float = 0.0
    last_update_step: int = -1

    def __post_init__(self) -> None:
        self.coefficients = np.zeros(self.dimension, dtype=np.float64)
        self.covariance = np.eye(self.dimension, dtype=np.float64) * 4.0

    def predict(self, context: np.ndarray) -> tuple[float, float]:
        mean = float(context @ self.coefficients)
        leverage = max(0.0, float(context @ self.covariance @ context))
        residual_sigma = math.sqrt(max(0.0, self.residual_variance))
        # We decide using uncertainty in the expected response, not a
        # prediction interval for one noisy execution. Including the
        # irreducible ``+1`` term made a small, consistently beneficial arm
        # impossible to prove even after repeated observations.
        uncertainty = residual_sigma * math.sqrt(leverage)
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


@dataclass
class RegretWindowState:
    entries: deque[tuple[float, float]] = field(default_factory=deque)
    queue_bad_streak: int = 0

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


@dataclass
class ContextRegimeState:
    ewma_context: tuple[float, ...] | None = None
    stable_observations: int = 0
    change_streak: int = 0
    warming_up: bool = False


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
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AdaptiveUBatchDecision":
        bucket_key = payload.get("bucket_key")
        if bucket_key is not None:
            bucket_key = tuple(bucket_key)
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
) -> tuple[int, int, int, int, int, int, int, float, float]:
    tokens = np.asarray(num_scheduled_tokens, dtype=np.int64)
    if tokens.size == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0
    total = int(tokens.sum())
    max_query = int(tokens.max())
    num_reqs = int(tokens.size)
    prefill_tokens = int(np.maximum(tokens - 1, 0).sum())
    decode_tokens = max(0, total - prefill_tokens)
    prefill_reqs = int(np.count_nonzero(tokens > 1))
    decode_reqs = max(0, num_reqs - prefill_reqs)
    prefill_ratio = prefill_tokens / total if total > 0 else 0.0
    avg_tokens_per_req = total / num_reqs if num_reqs > 0 else 0.0
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
        self._context_regimes: dict[
            tuple[str, str, str], ContextRegimeState
        ] = {}
        self._pending_contextual_outcome: PendingContextualOutcome | None = None
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
            self._contextual_cost[m] = state
        return state

    def _regret_state(self, bucket: WorkloadBucket) -> RegretWindowState:
        key = bucket.as_tuple()
        state = self._regret_windows.get(key)
        if state is None:
            state = RegretWindowState()
            self._regret_windows[key] = state
        return state

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
        log_scale, uncertainty = state.predict(context)
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
        if state.count == 0:
            relative_ratio = score.robust_cost_ms / max(
                safe_score.robust_cost_ms,
                1e-6,
            )
            log_uncertainty = 0.0
        else:
            log_ratio, log_uncertainty = state.predict(context)
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
        if state.count < min_observations:
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
        if state.ewma_context is None:
            state.ewma_context = context
            state.stable_observations = 1
            return False, 0.0
        distance = _context_distance(state.ewma_context, context)
        threshold = float(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_context_change_threshold",
                0.12,
            )
        )
        alpha = self._alpha()
        if distance > threshold:
            state.change_streak += 1
            # Stability is local to this contextual bucket.  An intervening
            # decode/other bucket must not erase prior stable visits, but a
            # material change within this bucket must make it safe again.
            state.stable_observations = 0
            state.warming_up = True
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
            if state.change_streak >= confirmations:
                state.ewma_context = context
                state.stable_observations = 1
                state.change_streak = 0
        else:
            state.change_streak = 0
            state.ewma_context = tuple(
                alpha * current + (1.0 - alpha) * previous
                for previous, current in zip(state.ewma_context, context)
            )
            state.stable_observations += 1
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
            if state.stable_observations >= stable_required:
                state.warming_up = False
        return (state.warming_up, distance)

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
        # Candidate-level cooldown handles a bad arm.  The bucket-level
        # budget receives a winsorized contribution, so one outlier pauses
        # exploration but cannot starve every other arm for the entire run.
        regret_cap_ms = (
            pending.baseline_ms * max_exploration_regret_pct / 100.0
        )
        regret_ms = min(raw_regret_ms, regret_cap_ms)
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
        regret_state.add(
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
        if (
            pending.selected_m != safe_m
            and raw_regret_pct > max_exploration_regret_pct
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
            regret_state.queue_bad_streak = 0
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
            "candidate_cooldown_until": candidate_cooldown_until,
            "queue_growth": queue_growth,
            "waiting_growth": waiting_growth,
            "queue_regressed": queue_regressed,
            "contextual_count": state.count,
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
        safe_cost, _, safe_count = self._predict_safe_contextual_cost(
            score=safe_score, context=context_array
        )
        regret_pct = self._regret_state(bucket).regret_pct()
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
        for score in candidate_scores:
            if score.m == safe_score.m or score.rejected:
                continue
            ratio, uncertainty, count = self._predict_relative_candidate(
                score=score,
                safe_score=safe_score,
                context=context_array,
            )
            upper_ratio = ratio * math.exp(risk_kappa * uncertainty)
            gain_lcb_pct = (1.0 - upper_ratio) * 100.0
            point_gain_pct = (1.0 - ratio) * 100.0
            evaluations[score.m] = {
                "contextual_relative_ratio": ratio,
                "contextual_log_uncertainty": uncertainty,
                "contextual_gain_lcb_pct": gain_lcb_pct,
                "contextual_point_gain_pct": point_gain_pct,
                "contextual_count": count,
            }
            if count >= min_observations and gain_lcb_pct >= min_gain:
                proven.append((gain_lcb_pct, score))
                continue
            prior_regret_pct = (
                (score.robust_cost_ms - safe_score.robust_cost_ms)
                / max(safe_score.robust_cost_ms, 1e-6)
                * 100.0
            )
            observed_candidate_is_plausible = (
                count < min_observations
                or point_gain_pct >= min_gain
            )
            if (
                observed_candidate_is_plausible
                and prior_regret_pct <= max_exploration_regret
            ):
                # Keep the analytical policy as the eligibility prior. Among
                # uncertain arms, collect evidence for the least-observed and
                # lowest-risk M first; M4 remains independently eligible after
                # M2 has received a real (non-compilation) observation.
                exploratory.append((count, -point_gain_pct, score))

        if regime_warmup:
            return (
                safe_score,
                "contextual_regime_warmup",
                None,
                safe_cost,
                regret_pct,
                evaluations,
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
        if regret_pct >= budget_pct:
            return (
                safe_score,
                "contextual_regret_budget",
                None,
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
        queue_stable = self._regret_state(bucket).queue_bad_streak == 0
        bucket_stable_observations = self._regime_state(
            bucket
        ).stable_observations
        can_explore = (
            bool(exploratory)
            and queue_stable
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
            return (
                selected,
                "contextual_exploration",
                evaluations[selected.m]["contextual_gain_lcb_pct"],
                safe_cost,
                regret_pct,
                evaluations,
            )
        if proven:
            gain_lcb_pct, selected = max(proven, key=lambda item: item[0])
            return (
                selected,
                "contextual_proven_gain",
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
    ) -> AdaptiveUBatchDecision:
        start_ns = time.perf_counter_ns()
        with self._lock:
            self._step_id += 1
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
                )
                self._write_trace({
                    "type": "adaptive_ubatch_ineligible",
                    "step_id": self._step_id,
                    "bucket": bucket.as_tuple(),
                    "reason": decision.reason,
                    "selected_m": 1,
                    "affects_cooldown": False,
                    "features": {
                        "total_tokens": features.total_tokens,
                        "num_reqs": features.num_reqs,
                        "max_query_len": features.max_query_len,
                        "avg_tokens_per_req": (
                            features.avg_tokens_per_req
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
                    }
                    for score in scores
                ),
                queue_depth=queue_depth,
                waiting_reqs=waiting_reqs,
                context_vector=context if contextual_mode else (),
                contextual_baseline_ms=contextual_baseline_ms,
                contextual_gain_lcb_pct=contextual_gain_lcb_pct,
                contextual_regret_pct=contextual_regret_pct,
            )
            self._write_trace({
                "type": "adaptive_ubatch_decision",
                "step_id": self._step_id,
                "bucket": bucket.as_tuple(),
                "features": {
                    "total_tokens": features.total_tokens,
                    "num_reqs": features.num_reqs,
                    "max_query_len": features.max_query_len,
                    "avg_tokens_per_req": features.avg_tokens_per_req,
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
            if (
                decision_reason
                in {"candidate_calibration", "contextual_exploration"}
                and has_alternative
                and cold_key not in self._discarded_cold_samples
            ):
                self._discarded_cold_samples.add(cold_key)
                self._write_trace({
                    "type": "adaptive_ubatch_cold_sample_ignored",
                    "step_id": self._step_id,
                    "bucket": bucket.as_tuple(),
                    "selected_m": m,
                    "actual_ms": actual_ms,
                    "predicted_ms": predicted_ms,
                    "reason": "first_bucket_m_sample_may_include_compilation",
                    "affects_cooldown": False,
                })
                return
            error_ms = actual_ms - predicted_ms
            alpha = self._alpha()
            prior_state_count = self._lookup_state(bucket, m).count
            selected_state = self._state(bucket, m)
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
                    self._pending_contextual_outcome = PendingContextualOutcome(
                        bucket_key=bucket.as_tuple(),
                        selected_m=m,
                        context=decision.context_vector,
                        prior_ms=prior_ms,
                        actual_ms=actual_ms,
                        baseline_ms=float(baseline_ms),
                        queue_depth=decision.queue_depth,
                        waiting_reqs=decision.waiting_reqs,
                    )
            degradation_pct = error_ms / max(predicted_ms, 1e-6) * 100.0
            bad_threshold = float(
                getattr(self.parallel_config, "adaptive_ubatch_bad_threshold_pct", 8.0)
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
                and enough_observations
                and degradation_pct > bad_threshold
            )
            # In contextual-safe mode the relative M=1 model and rolling
            # regret budget are authoritative. Absolute prior error is still
            # learned by the proposal model, but must not trigger a second,
            # conflicting safety policy.
            bad_execution = prediction_bad_execution and not contextual_mode
            bad_non_safe = bad_execution or bad_vs_safe
            if bad_non_safe and m != safe_m:
                bucket_state.bad_streak += 1
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
                        if bad_vs_safe
                        else int(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_cooldown_steps",
                                8,
                            )
                        )
                    )
                )
                if bad_vs_safe:
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
            self._write_trace({
                "type": "adaptive_ubatch_runtime_failure",
                "step_id": self._step_id,
                "bucket": bucket.as_tuple(),
                "selected_m": m,
                "fallback_m": self._current_m,
                "reason": reason,
                "affects_cooldown": affects_cooldown,
            })
