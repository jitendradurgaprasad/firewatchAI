"""Deterministic, explainable review-priority scoring for FIRMS observations."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CORE_WEIGHTS = {"recency": 0.20, "confidence": 0.25, "frp": 0.25,
                "repeat": 0.25, "completeness": 0.05}
# Reserve 15% for a match to FSI's forest-area-filtered alert points. Core
# weights retain their original relative proportions. With no imported context,
# active core weights renormalize to the exact former score.
DEFAULT_WEIGHTS = {**{key: value * 0.85 for key, value in CORE_WEIGHTS.items()},
                   "forest_context": 0.15}
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
    parsed_times = pd.to_datetime(frame["acq_datetime"], utc=True, errors="coerce")
    times = parsed_times.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    window_ns = int(window_hours * 3600 * 1_000_000_000)
    output = []
    missing_time = np.iinfo(np.int64).min
    for start in range(0, len(xyz), 2048):
        stop = min(start + 2048, len(xyz))
        neighbors = tree.query_ball_point(xyz[start:stop], radius_chord)
        for offset, ids in enumerate(neighbors):
            idx = start + offset
            candidates = np.asarray(ids, dtype=np.int64)
            candidates = candidates[candidates != idx]
            if times[idx] == missing_time:
                output.append(0)
            else:
                candidate_times = times[candidates]
                candidate_times = candidate_times[candidate_times != missing_time]
                output.append(int(np.count_nonzero(np.abs(candidate_times - times[idx]) <= window_ns)))
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
    minimum_coverage: float = 0.60,
    medium_threshold: float = 45.0,
    high_threshold: float = 70.0,
) -> pd.DataFrame:
    """Return a copy with factor scores, coverage, category, and explanation.

    Repeated detections are a review cue, not independent confirmation. FRP and
    confidence are related sensor signals; together their default share is
    capped at 50% of the score. Missing factors are omitted and remaining
    weights are renormalized; low coverage yields INSUFFICIENT INFORMATION.
    """
    if half_life_hours <= 0 or repeat_radius_km <= 0 or repeat_window_hours <= 0 or frp_cap_mw <= 0:
        raise ValueError("Scoring time, distance, and FRP scales must be positive.")
    if not 0 <= minimum_coverage <= 1 or not 0 <= medium_threshold < high_threshold <= 100:
        raise ValueError("Coverage must be 0–1 and thresholds must satisfy 0 ≤ medium < high ≤ 100.")
    if weights is None:
        active_weights = dict(DEFAULT_WEIGHTS)
    elif set(weights) == set(CORE_WEIGHTS):
        # Backward compatibility for saved settings from before forest context.
        active_weights = {**{key: float(value) * 0.85 for key, value in weights.items()},
                          "forest_context": 0.15}
    else:
        active_weights = dict(weights)
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
                    "missing_factors", "repeat_count", "review_status", "contributing_factors",
                    "data_quality_warnings", "forest_context", "forest_context_matches",
                    "forest_context_summary"):
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

    scores, categories, coverages, explanations, missings, contributions_by_row = [], [], [], [], [], []
    for i in range(len(result)):
        core_factors = {"recency": float(recency.iloc[i]) if pd.notna(recency.iloc[i]) else None,
                   "confidence": confidence[i], "frp": frp_values[i],
                   "repeat": repeat_score[i]}
        present_count = sum(val is not None for val in core_factors.values())
        factors = {**core_factors, "completeness": present_count / len(core_factors)}
        raw_forest_context = result.iloc[i].get("forest_context")
        if raw_forest_context is None or pd.isna(raw_forest_context):
            factors["forest_context"] = None
        else:
            factors["forest_context"] = 1.0 if bool(raw_forest_context) else 0.0
        missing = [key for key, val in factors.items() if val is None]
        available = {k: v for k, v in factors.items() if v is not None and active_weights[k] > 0}
        available_weight = sum(active_weights[k] for k in available)
        denominator = available_weight or 1.0
        # An unknown optional forest-context source does not count as a missing
        # measurement; coverage is relative to the factors that were configured.
        coverage_denominator = (sum(active_weights[k] for k in DEFAULT_WEIGHTS
                                    if k != "forest_context" and active_weights[k] > 0)
                                if factors["forest_context"] is None else 1.0)
        coverage = available_weight / coverage_denominator if coverage_denominator else 0.0
        score = 100 * sum(active_weights[k] * v for k, v in available.items()) / denominator
        if coverage < minimum_coverage or present_count < 2:
            category = "INSUFFICIENT INFORMATION"
        elif score >= high_threshold:
            category = "HIGH REVIEW PRIORITY"
        elif score >= medium_threshold:
            category = "MEDIUM REVIEW PRIORITY"
        else:
            category = "LOW REVIEW PRIORITY"
        contributions = [f"{k}: {100 * active_weights[k] * v / denominator:.1f} points"
                        for k, v in available.items()]
        contributions_by_row.append({k: round(active_weights[k] * v / denominator, 6)
                                     for k, v in available.items()})
        if missing:
            contributions.append("Not scored because missing: " + ", ".join(missing))
        if repeats[i]:
            contributions.append(f"{repeats[i]} nearby detection(s) within {repeat_radius_km:g} km and {repeat_window_hours:g} h")
        else:
            contributions.append(f"No nearby repeat in {repeat_radius_km:g} km / {repeat_window_hours:g} h window")
        if factors["forest_context"] is not None:
            match_count = int(result.iloc[i].get("forest_context_matches", 0) or 0)
            summary = result.iloc[i].get("forest_context_summary", "")
            contributions.append(f"Forest-area context: {match_count} spatial/time match(es)" +
                                 (f" ({summary})" if summary else "") +
                                 "; satellite alerts may share source detections and are not ground truth")
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
    result["contributing_factors"] = contributions_by_row
    result["data_quality_warnings"] = ["Missing score inputs: " + value if value else ""
                                       for value in missings]
    return result
