# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
import json
import math
import os
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any

import numpy as np
import regex as re


_BEST_POLICY_PROFILE = "best_20260729"


def _best_policy_profile_enabled() -> bool:
    """Use the controller behavior from the best July 29 experiment.

    The newer safety controller remains available for ablation/debugging via
    ``ADAPTIVE_UBATCH_POLICY_PROFILE=current_safe``.  Keep this switch outside
    ParallelConfig so the rollback does not change the public vLLM CLI or
    baseline execution path.
    """
    return (
        os.getenv("ADAPTIVE_UBATCH_POLICY_PROFILE", "current_safe")
        .strip()
        .lower()
        == _BEST_POLICY_PROFILE
    )


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
    waiting_count: int
    running_count: int
    oldest_wait_ms: float
    pending_first_token_count: int
    oldest_first_token_wait_ms: float
    pending_prefill_tokens: int
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
    pressure_count: int = 0
    ewma_queue_progress: float = 0.0
    ewma_first_token_rate: float = 0.0

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

    def update_queue_progress(
        self,
        *,
        queue_progress: float,
        first_token_rate: float,
        alpha: float,
    ) -> None:
        if self.pressure_count == 0:
            self.ewma_queue_progress = queue_progress
            self.ewma_first_token_rate = first_token_rate
        else:
            self.ewma_queue_progress = (
                alpha * queue_progress
                + (1.0 - alpha) * self.ewma_queue_progress
            )
            self.ewma_first_token_rate = (
                alpha * first_token_rate
                + (1.0 - alpha) * self.ewma_first_token_rate
            )
        self.pressure_count += 1


@dataclass
class BucketDecisionState:
    current_m: int
    last_change_step: int = -(10**9)
    pending_m: int | None = None
    pending_wins: int = 0
    bad_streak: int = 0
    queue_stall_streak: int = 0
    safe_refresh_required: bool = False
    validating_m: int | None = None
    probation_m: int | None = None
    probation_observations_left: int = 0


@dataclass(frozen=True)
class CandidateScore:
    m: int
    prior_cost_ms: float
    correction_ms: float
    calibrated_cost_ms: float
    uncertainty_ms: float
    robust_cost_ms: float
    count: int
    queue_progress: float | None = None
    first_token_rate: float | None = None
    queue_penalty_ms: float = 0.0
    pressure_count: int = 0
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
            "queue_progress": self.queue_progress,
            "first_token_rate": self.first_token_rate,
            "queue_penalty_ms": self.queue_penalty_ms,
            "pressure_count": self.pressure_count,
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
    waiting_count: int = 0
    running_count: int = 0
    oldest_wait_ms: float = 0.0
    pending_first_token_count: int = 0
    oldest_first_token_wait_ms: float = 0.0
    pending_prefill_tokens: int = 0
    probation: bool = False

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
            "waiting_count": self.waiting_count,
            "running_count": self.running_count,
            "oldest_wait_ms": self.oldest_wait_ms,
            "pending_first_token_count": self.pending_first_token_count,
            "oldest_first_token_wait_ms": self.oldest_first_token_wait_ms,
            "pending_prefill_tokens": self.pending_prefill_tokens,
            "probation": self.probation,
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
            waiting_count=max(0, int(payload.get("waiting_count", 0))),
            running_count=max(0, int(payload.get("running_count", 0))),
            oldest_wait_ms=max(
                0.0,
                float(payload.get("oldest_wait_ms", 0.0)),
            ),
            pending_first_token_count=max(
                0,
                int(payload.get("pending_first_token_count", 0)),
            ),
            oldest_first_token_wait_ms=max(
                0.0,
                float(payload.get("oldest_first_token_wait_ms", 0.0)),
            ),
            pending_prefill_tokens=max(
                0,
                int(payload.get("pending_prefill_tokens", 0)),
            ),
            probation=bool(payload.get("probation", False)),
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
        and decision.waiting_count == features.waiting_count
        and decision.running_count == features.running_count
        and decision.oldest_wait_ms == features.oldest_wait_ms
        and (
            decision.pending_first_token_count
            == features.pending_first_token_count
        )
        and (
            decision.oldest_first_token_wait_ms
            == features.oldest_first_token_wait_ms
        )
        and (
            decision.pending_prefill_tokens
            == features.pending_prefill_tokens
        )
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
        waiting_count=features.waiting_count,
        running_count=features.running_count,
        oldest_wait_ms=features.oldest_wait_ms,
        pending_first_token_count=features.pending_first_token_count,
        oldest_first_token_wait_ms=features.oldest_first_token_wait_ms,
        pending_prefill_tokens=features.pending_prefill_tokens,
        probation=decision.probation,
    )


def _queue_progress_score(
    decision: AdaptiveUBatchDecision,
    *,
    actual_ms: float,
    next_waiting_count: int | None,
    next_running_count: int | None,
    next_oldest_wait_ms: float | None,
    next_pending_first_token_count: int | None,
    next_oldest_first_token_wait_ms: float | None,
    next_pending_prefill_tokens: int | None,
    completed_first_token_count: int | None = None,
) -> tuple[float | None, bool, float]:
    """Return normalized request-pressure progress and a stall indicator.

    A score above zero means first-token/prefill pressure drained efficiently
    per unit wall time. A negative score means pressure accumulated. Count and
    token deltas must be time-normalized: every M executes the same scheduler
    output, so a per-step fraction alone would rate a slow M exactly like a
    fast M. Age remains a secondary signal because new arrivals can
    legitimately increase queue population between scheduler steps.
    """
    if (
        decision.pending_first_token_count <= 0
        or next_waiting_count is None
        or next_running_count is None
        or next_oldest_wait_ms is None
        or next_pending_first_token_count is None
        or next_oldest_first_token_wait_ms is None
        or next_pending_prefill_tokens is None
        or not _is_finite_positive(actual_ms)
    ):
        return None, False, 0.0

    next_waiting = max(0, int(next_waiting_count))
    next_running = max(0, int(next_running_count))
    next_oldest = max(0.0, float(next_oldest_wait_ms))
    next_first_token_count = max(
        0,
        int(next_pending_first_token_count),
    )
    next_first_token_wait = max(
        0.0,
        float(next_oldest_first_token_wait_ms),
    )
    next_prefill_tokens = max(0, int(next_pending_prefill_tokens))
    elapsed = max(actual_ms, 1e-6)
    oldest_progress = (
        decision.oldest_wait_ms + elapsed - next_oldest
    ) / elapsed
    waiting_progress = (
        decision.waiting_count - next_waiting
    ) / max(1, decision.waiting_count)
    running_progress = (
        decision.running_count - next_running
    ) / max(1, decision.running_count)
    first_token_age_progress = (
        decision.oldest_first_token_wait_ms
        + elapsed
        - next_first_token_wait
    ) / elapsed
    first_token_count_progress = (
        decision.pending_first_token_count
        - next_first_token_count
    ) / max(1, decision.pending_first_token_count)
    completed_first_tokens = (
        max(0, int(completed_first_token_count))
        if completed_first_token_count is not None
        else max(
            0,
            decision.pending_first_token_count
            - next_first_token_count,
        )
    )
    cohort_first_token_progress = (
        completed_first_tokens
        / max(1, decision.pending_first_token_count)
    )
    first_token_rate = completed_first_tokens * 1000.0 / elapsed
    prefill_progress = (
        decision.pending_prefill_tokens - next_prefill_tokens
    ) / max(1, decision.pending_prefill_tokens)
    def clamp(value: float) -> float:
        return max(-2.0, min(2.0, value))

    # Convert fractional progress to a per-second service efficiency before
    # clamping. This makes candidates comparable when they complete the same
    # scheduled work with different full-worker-step durations.
    rate_scale = 1000.0 / elapsed
    progress = (
        0.25 * clamp(cohort_first_token_progress * rate_scale)
        + 0.05 * clamp(first_token_count_progress * rate_scale)
        + 0.25 * clamp(prefill_progress * rate_scale)
        + 0.10 * clamp(waiting_progress * rate_scale)
        + 0.05 * clamp(running_progress * rate_scale)
        + 0.20 * clamp(first_token_age_progress)
        + 0.10 * clamp(oldest_progress)
    )
    stalled = (
        next_first_token_count >= decision.pending_first_token_count
        and next_first_token_wait
        >= decision.oldest_first_token_wait_ms + 0.75 * elapsed
        and next_prefill_tokens >= decision.pending_prefill_tokens
    )
    return progress, stalled, first_token_rate


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
    if _best_policy_profile_enabled():
        # Exact bucket boundaries used by the best-performing controller.
        # Pooling 3B/7B calibration avoids repeatedly re-calibrating nearly
        # identical serving shapes in a short 200-request experiment.
        if model_b > 7.0:
            model_bucket = "large"
        elif model_b >= 3.0:
            model_bucket = "medium"
        else:
            model_bucket = "small"
    elif model_b <= 4.5:
        model_bucket = "3b"
    elif model_b <= 10.0:
        model_bucket = "7b"
    elif model_b <= 20.0:
        model_bucket = "14b"
    else:
        model_bucket = "gt14b"

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
    waiting_count: int = 0,
    running_count: int = 0,
    oldest_wait_ms: float = 0.0,
    pending_first_token_count: int = 0,
    oldest_first_token_wait_ms: float = 0.0,
    pending_prefill_tokens: int = 0,
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
        waiting_count=max(0, int(waiting_count)),
        running_count=max(0, int(running_count)),
        oldest_wait_ms=max(0.0, float(oldest_wait_ms)),
        pending_first_token_count=max(
            0,
            int(pending_first_token_count),
        ),
        oldest_first_token_wait_ms=max(
            0.0,
            float(oldest_first_token_wait_ms),
        ),
        pending_prefill_tokens=max(0, int(pending_prefill_tokens)),
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


def _analytical_cost_ms(
    m: int,
    features: AdaptiveUBatchFeatures,
) -> float:
    params = _analytical_params_for_model(features.model_billions)
    total_tokens = max(1.0, float(features.total_tokens))
    decode_tokens = max(0.0, float(features.decode_tokens))
    m_float = max(1.0, float(m))
    compute = (
        params.alpha
        * total_tokens
        * (1.0 + (m_float * params.beta) /
           (total_tokens + m_float * params.gamma))
    )
    comm = m_float * params.delta + params.epsilon * total_tokens * features.hidden_size
    sample = params.sigma * decode_tokens
    step_ms = compute + comm + sample
    return step_ms / _effective_overlap(m, features)


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
    """Trace-calibrated risk-aware micro-batch degree controller.

    The analytical cost model remains the cold-start prior. Runtime
    observations update per-(bucket, M) EWMA prediction error and residual
    variance, allowing later decisions to use calibrated and risk-penalized
    costs while preserving an analytical-only mode for ablations.
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
                "calibrated_risk_aware",
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
        risk_enabled = mode == "calibrated_risk_aware"
        calibration_enabled = mode in {"calibrated", "calibrated_risk_aware"}
        risk_kappa = float(
            getattr(self.parallel_config, "adaptive_ubatch_risk_kappa", 1.0)
        )
        queue_guard_enabled = bool(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_enable_queue_guard",
                False,
            )
        )
        max_uncertainty_ratio = float(
            getattr(
                self.parallel_config,
                "adaptive_ubatch_max_uncertainty_ratio",
                0.15,
            )
        )
        safe_state = self._lookup_state(bucket, safe_m)
        pressure_target = max(
            2,
            int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_candidate_calibration_observations",
                    3,
                )
            ),
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
            scale = self._calibration_scale(state) if calibration_enabled else 1.0
            calibrated = max(1e-6, prior * scale)
            correction = calibrated - prior
            uncertainty = (
                self._uncertainty_ms(calibrated, state)
                if calibration_enabled
                else 0.0
            )
            robust = calibrated + (risk_kappa * uncertainty if risk_enabled else 0.0)
            queue_progress = (
                state.ewma_queue_progress
                if state.pressure_count > 0
                else None
            )
            first_token_rate = (
                state.ewma_first_token_rate
                if state.pressure_count > 0
                else None
            )
            queue_penalty_ms = 0.0
            if (
                queue_guard_enabled
                and m != safe_m
                and state.pressure_count >= pressure_target
            ):
                if safe_state.pressure_count >= pressure_target:
                    progress_regret = max(
                        0.0,
                        safe_state.ewma_queue_progress
                        - state.ewma_queue_progress,
                    )
                else:
                    # Before the safe reference is pressure-calibrated, only
                    # penalize a candidate that is actively accumulating
                    # queue pressure.
                    progress_regret = max(
                        0.0,
                        -state.ewma_queue_progress,
                    )
                queue_penalty_ms = calibrated * min(
                    0.50,
                    0.25 * progress_regret,
                )
                robust += queue_penalty_ms
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
                elif (
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
                queue_progress=queue_progress,
                first_token_rate=first_token_rate,
                queue_penalty_ms=queue_penalty_ms,
                pressure_count=state.pressure_count,
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
            if effective_m == int(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_safe_m",
                    1,
                )
            ):
                bucket_state.probation_m = None
                bucket_state.probation_observations_left = 0
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
                waiting_count=decision.waiting_count,
                running_count=decision.running_count,
                oldest_wait_ms=decision.oldest_wait_ms,
                pending_first_token_count=(
                    decision.pending_first_token_count
                ),
                oldest_first_token_wait_ms=(
                    decision.oldest_first_token_wait_ms
                ),
                pending_prefill_tokens=decision.pending_prefill_tokens,
                probation=decision.probation,
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
        waiting_count: int = 0,
        running_count: int = 0,
        oldest_wait_ms: float = 0.0,
        pending_first_token_count: int = 0,
        oldest_first_token_wait_ms: float = 0.0,
        pending_prefill_tokens: int = 0,
    ) -> AdaptiveUBatchDecision:
        start_ns = time.perf_counter_ns()
        with self._lock:
            self._step_id += 1
            features = extract_adaptive_ubatch_features(
                model_config=self.model_config,
                num_scheduled_tokens=num_scheduled_tokens,
                waiting_count=waiting_count,
                running_count=running_count,
                oldest_wait_ms=oldest_wait_ms,
                pending_first_token_count=pending_first_token_count,
                oldest_first_token_wait_ms=oldest_first_token_wait_ms,
                pending_prefill_tokens=pending_prefill_tokens,
            )
            bucket = WorkloadBucket.from_key(features.bucket_key)
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
                    waiting_count=features.waiting_count,
                    running_count=features.running_count,
                    oldest_wait_ms=features.oldest_wait_ms,
                    pending_first_token_count=(
                        features.pending_first_token_count
                    ),
                    oldest_first_token_wait_ms=(
                        features.oldest_first_token_wait_ms
                    ),
                    pending_prefill_tokens=features.pending_prefill_tokens,
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
                        "waiting_count": features.waiting_count,
                        "running_count": features.running_count,
                        "oldest_wait_ms": features.oldest_wait_ms,
                        "pending_first_token_count": (
                            features.pending_first_token_count
                        ),
                        "oldest_first_token_wait_ms": (
                            features.oldest_first_token_wait_ms
                        ),
                        "pending_prefill_tokens": (
                            features.pending_prefill_tokens
                        ),
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
                    waiting_count=features.waiting_count,
                    running_count=features.running_count,
                    oldest_wait_ms=features.oldest_wait_ms,
                    pending_first_token_count=(
                        features.pending_first_token_count
                    ),
                    oldest_first_token_wait_ms=(
                        features.oldest_first_token_wait_ms
                    ),
                    pending_prefill_tokens=features.pending_prefill_tokens,
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

            reason = "robust_cost_improvement"
            fallback = False
            if not valid_scores:
                selected = current_score
                reason = "all_candidates_rejected"
                fallback = True
            elif (
                not _best_policy_profile_enabled()
                and bucket_state.probation_m == previous_m
                and previous_m != safe_m
                and bucket_state.probation_observations_left > 0
            ):
                # Once promoted, measure the candidate on consecutive
                # comparable steps. Do not let unrelated candidate
                # calibration interrupt the safety observation window.
                selected = current_score
                reason = "candidate_probation"
            elif (
                not _best_policy_profile_enabled()
                and bucket_state.safe_refresh_required
            ):
                safe_score = next(
                    (
                        score
                        for score in valid_scores
                        if score.m == safe_m
                    ),
                    None,
                )
                selected = safe_score or current_score
                reason = "paired_safe_refresh"
            else:
                safe_score = next(
                    (
                        score
                        for score in valid_scores
                        if score.m == safe_m
                    ),
                    None,
                )
                revalidation_enabled = bool(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_enable_exploration",
                        False,
                    )
                )
                revalidation_interval = max(
                    1,
                    int(
                        getattr(
                            self.parallel_config,
                            "adaptive_ubatch_exploration_interval_steps",
                            64,
                        )
                        or 64
                    ),
                )
                safe_state = self._lookup_state(bucket, safe_m)
                safe_reference_stale = (
                    previous_m != safe_m
                    and safe_score is not None
                    and safe_state.count > 0
                    and (
                        self._step_id - safe_state.last_update_step
                        >= revalidation_interval
                    )
                )
                if (
                    not _best_policy_profile_enabled()
                    and revalidation_enabled
                    and safe_reference_stale
                ):
                    # Periodically pair an active M>1 policy with a current M=1
                    # sample. This detects QPS/load drift without permanently
                    # including QPS in the bucket key.
                    selected = safe_score
                    reason = "periodic_safe_refresh"
                    bucket_state.pending_m = None
                    bucket_state.pending_wins = 0
                else:
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
                if selected is not None:
                    pass
                elif (
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
                        pressure_ratio = features.waiting_count / max(
                            1,
                            features.num_reqs,
                        )
                        pressure_extra_gain = min(
                            6.0,
                            pressure_ratio * 2.0
                            + min(features.oldest_wait_ms / 1000.0, 2.0),
                        )
                        required_safe_gain = (
                            min_gain + pressure_extra_gain
                            if bool(
                                getattr(
                                    self.parallel_config,
                                    "adaptive_ubatch_enable_queue_guard",
                                    False,
                                )
                            )
                            and features.waiting_count > 0
                            and best.m != safe_m
                            else min_gain
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
                            and safe_gain_pct < required_safe_gain
                        ):
                            # The safe candidate is the explicit online
                            # reference. A non-safe M must beat it by the full
                            # switch margin; a marginal win is not enough to
                            # pay adaptive split/communication overhead.
                            selected = safe_score
                            reason = (
                                "queue_pressure_gain_guard"
                                if required_safe_gain > min_gain
                                else "safe_m_gain_guard"
                            )
                            bucket_state.pending_m = None
                            bucket_state.pending_wins = 0
                        else:
                            selected = best
                        if (
                            bool(
                                getattr(
                                    self.parallel_config,
                                    "adaptive_ubatch_enable_queue_guard",
                                    False,
                                )
                            )
                            and selected.m != safe_m
                            and safe_score is not None
                        ):
                            selected_state = self._lookup_state(
                                bucket,
                                selected.m,
                            )
                            safe_state = self._lookup_state(
                                bucket,
                                safe_m,
                            )
                            progress_target = max(
                                2,
                                calibration_target,
                            )
                            progress_tolerance = max(
                                0.02,
                                float(
                                    getattr(
                                        self.parallel_config,
                                        (
                                            "adaptive_ubatch_"
                                            "max_exploration_regret_pct"
                                        ),
                                        5.0,
                                    )
                                )
                                / 100.0,
                            )
                            if (
                                selected_state.pressure_count
                                >= progress_target
                                and safe_state.pressure_count
                                >= progress_target
                            ):
                                queue_regret = (
                                    safe_state.ewma_queue_progress
                                    - selected_state.ewma_queue_progress
                                )
                                rate_regret = 0.0
                                if safe_state.ewma_first_token_rate > 0:
                                    rate_regret = (
                                        safe_state.ewma_first_token_rate
                                        - selected_state.ewma_first_token_rate
                                    ) / max(
                                        safe_state.ewma_first_token_rate,
                                        1e-6,
                                    )
                                request_regressed = (
                                    rate_regret > progress_tolerance
                                    or queue_regret
                                    > 2.0 * progress_tolerance
                                )
                                if request_regressed:
                                    selected = safe_score
                                    reason = (
                                        "request_progress_promotion_guard"
                                    )
                                    self._cooldown_until[
                                        (bucket.as_tuple(), best.m)
                                    ] = (
                                        self._step_id
                                        + max(
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
                                    )
                                    bucket_state.pending_m = None
                                    bucket_state.pending_wins = 0
                        min_hold = int(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_min_hold_steps",
                                4,
                            )
                        )
                        if reason in {
                            "safe_m_gain_guard",
                            "queue_pressure_gain_guard",
                            "request_progress_promotion_guard",
                        }:
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
                if (
                    not _best_policy_profile_enabled()
                    and selected_m != safe_m
                    and reason
                    not in {
                        "candidate_calibration",
                        "controlled_exploration",
                        "paired_safe_refresh",
                    }
                ):
                    bucket_state.probation_m = selected_m
                    bucket_state.probation_observations_left = max(
                        2,
                        int(
                            getattr(
                                self.parallel_config,
                                (
                                    "adaptive_ubatch_"
                                    "exploration_stable_steps"
                                ),
                                8,
                            )
                        ),
                    )
                else:
                    bucket_state.probation_m = None
                    bucket_state.probation_observations_left = 0
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
                candidate_scores=tuple(s.to_payload() for s in scores),
                waiting_count=features.waiting_count,
                running_count=features.running_count,
                oldest_wait_ms=features.oldest_wait_ms,
                pending_first_token_count=(
                    features.pending_first_token_count
                ),
                oldest_first_token_wait_ms=(
                    features.oldest_first_token_wait_ms
                ),
                pending_prefill_tokens=features.pending_prefill_tokens,
                probation=(
                    bucket_state.probation_m == selected_m
                    and selected_m != safe_m
                    and bucket_state.probation_observations_left > 0
                ),
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
                    "waiting_count": features.waiting_count,
                    "running_count": features.running_count,
                    "oldest_wait_ms": features.oldest_wait_ms,
                    "pending_first_token_count": (
                        features.pending_first_token_count
                    ),
                    "oldest_first_token_wait_ms": (
                        features.oldest_first_token_wait_ms
                    ),
                    "pending_prefill_tokens": (
                        features.pending_prefill_tokens
                    ),
                },
                "previous_m": previous_m,
                "selected_m": selected_m,
                "predicted_gain_pct": decision.predicted_gain_pct,
                "switched": switched,
                "fallback": fallback,
                "reason": reason,
                "pending_m": bucket_state.pending_m,
                "pending_wins": bucket_state.pending_wins,
                "probation": decision.probation,
                "probation_observations_left": (
                    bucket_state.probation_observations_left
                ),
                "candidates": list(decision.candidate_scores),
                "decision_overhead_us": overhead_us,
            })
            return decision

    def observe(
        self,
        decision: AdaptiveUBatchDecision,
        *,
        forward_ms: float,
        next_waiting_count: int | None = None,
        next_running_count: int | None = None,
        next_oldest_wait_ms: float | None = None,
        next_pending_first_token_count: int | None = None,
        next_oldest_first_token_wait_ms: float | None = None,
        next_pending_prefill_tokens: int | None = None,
        completed_first_token_count: int | None = None,
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
            if (
                decision.reason == "candidate_calibration"
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
            (
                queue_progress,
                queue_stalled,
                first_token_rate,
            ) = _queue_progress_score(
                decision,
                actual_ms=actual_ms,
                next_waiting_count=next_waiting_count,
                next_running_count=next_running_count,
                next_oldest_wait_ms=next_oldest_wait_ms,
                next_pending_first_token_count=(
                    next_pending_first_token_count
                ),
                next_oldest_first_token_wait_ms=(
                    next_oldest_first_token_wait_ms
                ),
                next_pending_prefill_tokens=next_pending_prefill_tokens,
                completed_first_token_count=completed_first_token_count,
            )
            if queue_progress is not None:
                selected_state.update_queue_progress(
                    queue_progress=queue_progress,
                    first_token_rate=first_token_rate,
                    alpha=alpha,
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
            if (
                not _best_policy_profile_enabled()
                and m != safe_m
                and decision.reason == "candidate_calibration"
                and selected_state.count >= accepted_target
            ):
                bucket_state.safe_refresh_required = True
                bucket_state.validating_m = m
            elif (
                m == safe_m
                and bucket_state.safe_refresh_required
            ):
                bucket_state.safe_refresh_required = False
                bucket_state.validating_m = None
            relative_safe_regret_pct = None
            bad_vs_safe = False
            queue_progress_regret = None
            bad_queue_progress = False
            if (
                m != safe_m
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
            pressure_target = max(2, accepted_target)
            queue_guard_enabled = bool(
                getattr(
                    self.parallel_config,
                    "adaptive_ubatch_enable_queue_guard",
                    False,
                )
            )
            progress_regret_threshold = max(
                0.02,
                float(
                    getattr(
                        self.parallel_config,
                        "adaptive_ubatch_max_exploration_regret_pct",
                        5.0,
                    )
                )
                / 100.0,
            )
            if (
                queue_guard_enabled
                and m != safe_m
                and selected_state.pressure_count >= pressure_target
            ):
                if safe_state.pressure_count >= pressure_target:
                    queue_progress_regret = (
                        safe_state.ewma_queue_progress
                        - selected_state.ewma_queue_progress
                    )
                    first_token_rate_regret = 0.0
                    if safe_state.ewma_first_token_rate > 0:
                        first_token_rate_regret = (
                            safe_state.ewma_first_token_rate
                            - selected_state.ewma_first_token_rate
                        ) / max(
                            safe_state.ewma_first_token_rate,
                            1e-6,
                        )
                    bad_queue_progress = (
                        queue_progress_regret
                        > 2.0 * progress_regret_threshold
                        or first_token_rate_regret
                        > progress_regret_threshold
                    )
                else:
                    bad_queue_progress = (
                        selected_state.ewma_queue_progress
                        < -2.0 * progress_regret_threshold
                    )
            if queue_guard_enabled and queue_stalled and m != safe_m:
                bucket_state.queue_stall_streak += 1
            else:
                bucket_state.queue_stall_streak = 0
            repeated_queue_stall = bucket_state.queue_stall_streak >= 2
            enough_observations = prior_state_count >= self._min_observations()
            bad_execution = (
                has_alternative
                and enough_observations
                and degradation_pct > bad_threshold
            )
            bad_non_safe = (
                bad_execution
                or bad_vs_safe
                or bad_queue_progress
                or repeated_queue_stall
            )
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
                        if (
                            bad_vs_safe
                            or bad_queue_progress
                            or repeated_queue_stall
                        )
                        else int(
                            getattr(
                                self.parallel_config,
                                "adaptive_ubatch_cooldown_steps",
                                8,
                            )
                        )
                    )
                )
                if (
                    bad_vs_safe
                    or bad_queue_progress
                    or repeated_queue_stall
                ):
                    self._current_m = safe_m
                    bucket_state.current_m = safe_m
                    bucket_state.pending_m = None
                    bucket_state.pending_wins = 0
                    bucket_state.probation_m = None
                    bucket_state.probation_observations_left = 0
            else:
                bucket_state.bad_streak = 0
                if (
                    m != safe_m
                    and bucket_state.probation_m == m
                    and bucket_state.probation_observations_left > 0
                ):
                    bucket_state.probation_observations_left -= 1
                    if bucket_state.probation_observations_left <= 0:
                        bucket_state.probation_m = None
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
                "bad_execution": bad_execution,
                "relative_safe_regret_pct": relative_safe_regret_pct,
                "bad_vs_safe": bad_vs_safe,
                "queue_progress": queue_progress,
                "first_token_rate": first_token_rate,
                "completed_first_token_count": (
                    completed_first_token_count
                ),
                "queue_progress_ewma": (
                    selected_state.ewma_queue_progress
                    if selected_state.pressure_count > 0
                    else None
                ),
                "queue_progress_regret": queue_progress_regret,
                "progress_regret_threshold": (
                    progress_regret_threshold
                ),
                "queue_stalled": queue_stalled,
                "queue_stall_streak": bucket_state.queue_stall_streak,
                "bad_queue_progress": bad_queue_progress,
                "next_waiting_count": next_waiting_count,
                "next_running_count": next_running_count,
                "next_oldest_wait_ms": next_oldest_wait_ms,
                "next_pending_first_token_count": (
                    next_pending_first_token_count
                ),
                "next_oldest_first_token_wait_ms": (
                    next_oldest_first_token_wait_ms
                ),
                "next_pending_prefill_tokens": (
                    next_pending_prefill_tokens
                ),
                "has_alternative": has_alternative,
                "enough_observations": enough_observations,
                "affects_cooldown": (
                    bad_non_safe and m != safe_m
                ),
                "probation": decision.probation,
                "probation_observations_left": (
                    bucket_state.probation_observations_left
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
            bucket_state.probation_m = None
            bucket_state.probation_observations_left = 0
            self._write_trace({
                "type": "adaptive_ubatch_runtime_failure",
                "step_id": self._step_id,
                "bucket": bucket.as_tuple(),
                "selected_m": m,
                "fallback_m": self._current_m,
                "reason": reason,
                "affects_cooldown": affects_cooldown,
            })
