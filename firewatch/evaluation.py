"""Controlled synthetic functional checks; never accuracy or wildfire validation."""
from __future__ import annotations

import pandas as pd

from .association import associate_observations, haversine_km
from .ranking import DEFAULT_WEIGHTS, score_observations


def run_functional_evaluation() -> dict:
    base = pd.DataFrame([
        {"observation_id": "synthetic-high", "latitude": 10.0, "longitude": 20.0,
         "acq_datetime": "2026-01-01T12:00:00Z", "confidence": "high", "frp": 150.0},
        {"observation_id": "synthetic-low", "latitude": 30.0, "longitude": 40.0,
         "acq_datetime": "2025-12-01T12:00:00Z", "confidence": "low", "frp": 0.2},
        {"observation_id": "synthetic-missing", "latitude": -30.0, "longitude": -80.0,
         "acq_datetime": "2026-01-01T12:00:00Z", "confidence": None, "frp": None},
        {"observation_id": "synthetic-repeat-a", "latitude": 0.0, "longitude": 0.0,
         "acq_datetime": "2026-01-01T00:00:00Z", "confidence": "nominal", "frp": 2.0},
        {"observation_id": "synthetic-repeat-b", "latitude": 0.0, "longitude": 0.001,
         "acq_datetime": "2026-01-01T01:00:00Z", "confidence": "nominal", "frp": 2.0},
        {"observation_id": "synthetic-late", "latitude": 0.0, "longitude": 0.001,
         "acq_datetime": "2026-01-02T12:00:00Z", "confidence": "nominal", "frp": 2.0},
    ])
    reference = "2026-01-01T12:00:00Z"
    scored = score_observations(base, reference_time=reference)
    rerun = score_observations(base, reference_time=reference)
    clustered, summaries = associate_observations(base, distance_km=1, time_window_hours=6)
    weight_variant = score_observations(
        base.iloc[:2], weights={"recency": .1, "confidence": .1, "frp": .7, "repeat": .05,
                                "completeness": .05}, reference_time=reference)
    checks = {
        "deterministic_scores": scored["priority_score"].tolist() == rerun["priority_score"].tolist(),
        "stronger_signal_ranks_higher": scored.iloc[0]["priority_score"] > scored.iloc[1]["priority_score"],
        "missing_signals_explained": "confidence, frp" in scored.iloc[2]["missing_factors"],
        "missing_signal_coverage_abstains": scored.iloc[2]["priority_category"] == "INSUFFICIENT INFORMATION",
        "score_range_is_bounded": bool(scored["priority_score"].between(0, 100).all()),
        "weight_configuration_changes_score": weight_variant.iloc[0]["priority_score"] != scored.iloc[0]["priority_score"],
        "default_weights_sum_to_one": abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9,
        "repeat_time_gate_uses_elapsed_hours": scored.set_index("observation_id")["repeat_count"].to_dict() == {
            "synthetic-high":0, "synthetic-low":0, "synthetic-missing":0,
            "synthetic-repeat-a":1, "synthetic-repeat-b":1, "synthetic-late":0},
        "association_respects_both_distance_and_time": (
            clustered.iloc[3]["association_id"] == clustered.iloc[4]["association_id"] and
            pd.isna(clustered.iloc[5]["association_id"])),
        "great_circle_distance_units_are_km": abs(haversine_km(0, 0, 0, 1) - 111.195) < .02,
        "cluster_summary_is_not_empty_for_synthetic_pair": not summaries.empty and int(summaries.iloc[0]["observation_count"]) == 2,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"evaluation_type": "synthetic functional rule checks",
            "synthetic_records": len(base), "checks_passed": int(sum(checks.values())),
            "checks_total": len(checks), "checks": checks,
            "interpretation": "This verifies code behavior only; it is not an accuracy evaluation and uses no real-world fire labels."}
