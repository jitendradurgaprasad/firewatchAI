"""Functional self-checks for implementation rules, not fire accuracy metrics."""

from __future__ import annotations

import pandas as pd

from .ranking import DEFAULT_WEIGHTS, score_observations


def run_functional_evaluation() -> dict:
    base = pd.DataFrame([
        {"observation_id": "synthetic-high", "latitude": 10.0, "longitude": 20.0,
         "acq_datetime": "2026-01-01T12:00:00Z", "confidence": "high", "frp": 150.0},
        {"observation_id": "synthetic-low", "latitude": 30.0, "longitude": 40.0,
         "acq_datetime": "2025-12-01T12:00:00Z", "confidence": "low", "frp": 0.2},
        {"observation_id": "synthetic-missing", "latitude": -30.0, "longitude": -80.0,
         "acq_datetime": "2026-01-01T12:00:00Z", "confidence": None, "frp": None},
    ])
    scored = score_observations(base, reference_time="2026-01-01T12:00:00Z")
    rerun = score_observations(base, reference_time="2026-01-01T12:00:00Z")
    checks = {
        "deterministic_scores": scored["priority_score"].tolist() == rerun["priority_score"].tolist(),
        "stronger_signal_ranks_higher": scored.iloc[0]["priority_score"] > scored.iloc[1]["priority_score"],
        "missing_signals_explained": "confidence, frp" in scored.iloc[2]["missing_factors"],
        "missing_signal_coverage_abstains": scored.iloc[2]["priority_category"] == "INSUFFICIENT INFORMATION",
        "score_range_is_bounded": bool(scored["priority_score"].between(0, 100).all()),
        "weight_configuration_changes_score": score_observations(
            base.iloc[:2], weights={"recency": .1, "confidence": .1, "frp": .7, "repeat": .05,
                                   "completeness": .05},
            reference_time="2026-01-01T12:00:00Z").iloc[0]["priority_score"] != scored.iloc[0]["priority_score"],
        "declared_default_weights_sum_to_one": abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9,
    }
    return {"evaluation_type": "synthetic functional rule checks",
            "synthetic_records": len(base), "checks_passed": sum(checks.values()),
            "checks_total": len(checks), "checks": checks,
            "interpretation": "This verifies code behavior only; it is not an accuracy evaluation and uses no real-world fire labels."}
