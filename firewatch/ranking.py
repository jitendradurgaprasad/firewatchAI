"""Deterministic, explainable review-priority scoring for FIRMS observations."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_WEIGHTS = {"recency": 0.20, "confidence": 0.25, "frp": 0.25,
                   "repeat": 0.25, "completeness": 0.05}
CONFIDENCE_SCORE = {"low": 0.30, "nominal": 0.65, "high": 0.90}
EARTH_RADIUS_KM = 6371.0088


def _confidence_value(value):
    if value is None or pd.isna(value) or value == "":
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in CONFIDENCE_SCORE:
            return CONFIDENCE_SCORE[value]
    try:
        numeric = float(value)
        return numeric / 100 if math.isfinite(numeric) and 0 <= numeric <= 100 else None
    except (TypeError, ValueError):
        return None


def _repeat_counts(frame: pd.DataFrame, radius_km: float, window_hours: float) -> list[int]:
    if frame.empty:
        return []
    coords = np.radians(frame[["latitude", "longitude"]].astype(float).to_numpy())
    xyz = np.column_stack((np.cos(coords[:, 0]) * np.cos(coords[:, 1]),
                           np.cos(coords[:, 0]) * np.sin(coords[:, 1]), np.sin(coords[:, 0])))
    radius_chord = 2 * math.sin(min(radius_km, math.pi * EARTH_RADIUS_KM) /
                                (2 * EARTH_RADIUS_KM))
    tree = cKDTree(xyz)
    times = pd.to_datetime(frame["acq_datetime"], utc=True, errors="coerce").astype("int64").to_numpy()
    window_ns = int(window_hours * 3600 * 1_000_000_000)
    neighbors = tree.query_ball_point(xyz, radius_chord)
    output = []
    for idx, ids in enumerate(neighbors):
        output.append(sum(1 for other in ids if other != idx and
                          times[idx] != np.iinfo(np.int64).min and
                          times[other] != np.iinfo(np.int64).min and
                          abs(int(times[idx]) - int(times[other])) <= window_ns))
    return output


def score_observations(
    observations: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
    reference_time: datetime | pd.Timestamp | None = None,
    half_life_hours: float = 72.0,
    repeat_radius_km: float = 1.0,
    repeat_window_hours: float = 24.0,
    frp_cap_mw: float = 500.0,
) -> pd.DataFrame:
    """Return a copy with factor scores, coverage, category, and explanation.

    Repeated detections are a review cue, not independent confirmation. FRP and
    confidence are related sensor signals; together their default share is
    capped at 50% of the score. Missing factors are omitted and remaining
    weights are renormalized; low coverage yields INSUFFICIENT INFORMATION.
    """
    if half_life_hours <= 0 or repeat_radius_km <= 0 or repeat_window_hours <= 0 or frp_cap_mw <= 0:
        raise ValueError("Scoring time, distance, and FRP scales must be positive.")
    active_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(active_weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"Weights must contain exactly: {', '.join(DEFAULT_WEIGHTS)}")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in active_weights.values()):
        raise ValueError("Weights must be finite and non-negative.")
    weight_total = sum(float(v) for v in active_weights.values())
    if weight_total <= 0:
        raise ValueError("At least one ranking weight must be positive.")
    active_weights = {k: float(v) / weight_total for k, v in active_weights.items()}

    result = observations.copy()
    if result.empty:
        for col in ("priority_score", "priority_category", "score_coverage", "explanation",
                    "missing_factors", "repeat_count", "review_status"):
            if col not in result:
                result[col] = pd.Series(dtype="object")
        return result
    dates = pd.to_datetime(result["acq_datetime"], utc=True, errors="coerce")
    if reference_time is None:
        reference = dates.max()
    else:
        reference = pd.Timestamp(reference_time)
        reference = reference.tz_localize("UTC") if reference.tzinfo is None else reference.tz_convert("UTC")
    if pd.isna(reference):
        reference = pd.Timestamp(datetime.now(timezone.utc))
    age_hours = (reference - dates).dt.total_seconds().div(3600).clip(lower=0)
    recency = np.exp2(-age_hours / half_life_hours)
    repeats = _repeat_counts(result, repeat_radius_km, repeat_window_hours)
    repeat_score = [min(n / 5, 1.0) for n in repeats]
    confidence = [_confidence_value(value) for value in result.get("confidence", pd.Series([None] * len(result), index=result.index))]
    frp_values = []
    for value in result.get("frp", pd.Series([None] * len(result), index=result.index)):
        try:
            value = float(value)
            frp_values.append(min(math.log1p(max(value, 0)) / math.log1p(frp_cap_mw), 1.0)
                              if math.isfinite(value) else None)
        except (TypeError, ValueError):
            frp_values.append(None)

    scores, categories, coverages, explanations, missings = [], [], [], [], []
    for i in range(len(result)):
        factors = {"recency": float(recency.iloc[i]) if pd.notna(recency.iloc[i]) else None,
                   "confidence": confidence[i], "frp": frp_values[i],
                   "repeat": repeat_score[i]}
        missing = [key for key, val in factors.items() if val is None]
        present_count = sum(val is not None for val in factors.values())
        factors["completeness"] = present_count / len(factors)
        available = {k: v for k, v in factors.items() if v is not None and active_weights[k] > 0}
        coverage = sum(active_weights[k] for k in available)
        denominator = coverage or 1.0
        score = 100 * sum(active_weights[k] * v for k, v in available.items()) / denominator
        if coverage < 0.60 or present_count < 2:
            category = "INSUFFICIENT INFORMATION"
        elif score >= 70:
            category = "HIGH REVIEW PRIORITY"
        elif score >= 45:
            category = "MEDIUM REVIEW PRIORITY"
        else:
            category = "LOW REVIEW PRIORITY"
        contributions = [f"{k}: {100 * active_weights[k] * v / denominator:.1f} points"
                        for k, v in available.items()]
        if missing:
            contributions.append("Not scored because missing: " + ", ".join(missing))
        if repeats[i]:
            contributions.append(f"{repeats[i]} nearby detection(s) within {repeat_radius_km:g} km and {repeat_window_hours:g} h")
        else:
            contributions.append(f"No nearby repeat in {repeat_radius_km:g} km / {repeat_window_hours:g} h window")
        contributions.append("Review priority only; this is not a fire-severity or confirmation score.")
        scores.append(round(score, 2))
        categories.append(category)
        coverages.append(round(coverage, 4))
        explanations.append("; ".join(contributions))
        missings.append(", ".join(missing))
    result["priority_score"] = scores
    result["priority_category"] = categories
    result["score_coverage"] = coverages
    result["explanation"] = explanations
    result["missing_factors"] = missings
    result["repeat_count"] = repeats
    return result
