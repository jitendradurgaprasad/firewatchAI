"""Deterministic, explicitly synthetic forest-fire triage demo data.

These generated records exercise ingestion, forest-context matching, ranking,
event grouping and review. They are not observations, ground truth, or model
training data for real-world performance claims.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


OBSERVATION_FIELDS = (
    "latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "frp", "bright_ti4", "bright_ti5", "daynight", "scan", "track",
    "version", "type", "synthetic_truth", "synthetic_event_id", "synthetic_scenario",
)
CONTEXT_FIELDS = ("Fire Date", "Time", "Source", "Lat", "Lon", "State", "District", "Division", "Range", "Beat")
DEMO_SEED = 20261016


@dataclass(frozen=True)
class SyntheticForestDemo:
    observations_csv: bytes
    context_csv: bytes
    manifest: dict


def _csv_bytes(fields: tuple[str, ...], rows: list[dict], *, excel_headers: bool = False) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig" if excel_headers else "utf-8")


def generate_synthetic_forest_demo(seed: int = DEMO_SEED) -> SyntheticForestDemo:
    """Return 800 synthetic FIRMS-shaped rows and correlated forest context.

    The coordinates are generated scenario coordinates within India's broad
    extent, not verified forests or observed fire locations. Labels are known
    because this program creates them.
    """
    rng = np.random.default_rng(seed)
    # Twenty positive and twenty negative events, 20 rows per event.
    scenarios = []
    for i in range(16):
        scenarios.append(("forest_fire", "simulated_forest_fire", i % 8))
    for i in range(8):
        scenarios.append(("non_fire", "simulated_forest_false_alarm", (i + 3) % 8))
    for i in range(16):
        scenario = "simulated_industrial_heat" if i < 8 else "simulated_agricultural_burn"
        scenarios.append(("non_fire", scenario, (i + 5) % 8))

    # Synthetic regional centers only; no forest-boundary data is consulted.
    centers = [(28.2, 84.5), (26.1, 92.7), (23.8, 85.0), (21.7, 83.1),
               (19.4, 81.2), (17.8, 80.8), (25.4, 91.2), (12.9, 75.6)]
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    observations: list[dict] = []
    context: list[dict] = []
    class_counts = {"forest_fire": 0, "non_fire": 0}

    for event_index, (truth, scenario, center_index) in enumerate(scenarios):
        event_id = f"SYN-{seed}-{event_index + 1:03d}"
        center_lat, center_lon = centers[center_index]
        start_time = start + timedelta(days=event_index % 8, hours=(event_index * 3) % 16)
        for detection_index in range(20):
            lat = center_lat + float(rng.normal(0, 0.0035))
            lon = center_lon + float(rng.normal(0, 0.0035))
            timestamp = start_time + timedelta(minutes=detection_index * 15)
            if scenario == "simulated_forest_fire":
                confidence = str(rng.choice(["high", "nominal"], p=[0.65, 0.35]))
                frp = float(rng.lognormal(2.0, 0.8))
            elif scenario == "simulated_forest_false_alarm":
                confidence = str(rng.choice(["high", "nominal", "low"], p=[0.35, 0.45, 0.20]))
                frp = float(rng.lognormal(2.0, 0.9))
            elif scenario == "simulated_industrial_heat":
                confidence = str(rng.choice(["high", "nominal"], p=[0.55, 0.45]))
                frp = float(rng.lognormal(3.7, 0.65))
            else:
                confidence = str(rng.choice(["nominal", "low"], p=[0.65, 0.35]))
                frp = float(rng.lognormal(1.5, 0.7))
            satellite = str(rng.choice(["NOAA-20", "NOAA-21", "SNPP"]))
            common = {"latitude": f"{lat:.6f}", "longitude": f"{lon:.6f}",
                      "acq_date": timestamp.strftime("%Y-%m-%d"), "acq_time": timestamp.strftime("%H%M"),
                      "satellite": satellite, "instrument": "VIIRS", "confidence": confidence,
                      "frp": f"{frp:.2f}", "bright_ti4": f"{310 + min(frp, 120) * .35:.2f}",
                      "bright_ti5": f"{285 + min(frp, 100) * .20:.2f}", "daynight": "D" if 6 <= timestamp.hour < 18 else "N",
                      "scan": "0.40", "track": "0.40", "version": "2.0NRT", "type": "0",
                      "synthetic_truth": truth, "synthetic_event_id": event_id,
                      "synthetic_scenario": scenario}
            observations.append(common)
            class_counts[truth] += 1

            # Forest context is generated for true forest-fire scenarios and
            # some deliberately false forest-area anomalies. It is correlated
            # with the synthetic hotspot, just like a real FSI satellite alert.
            if scenario in {"simulated_forest_fire", "simulated_forest_false_alarm"}:
                ctx_lat = lat + float(rng.normal(0, 0.0005))
                ctx_lon = lon + float(rng.normal(0, 0.0005))
                ctx_time = timestamp + timedelta(minutes=int(rng.integers(-8, 9)))
                context.append({"Fire Date": ctx_time.strftime("%Y-%m-%d"),
                                "Time": ctx_time.strftime("%H:%M:%S"),
                                "Source": "SYNTHETIC-VIIRS", "Lat": f"{ctx_lat:.6f}",
                                "Lon": f"{ctx_lon:.6f}", "State": "SIMULATED",
                                "District": "SYNTHETIC", "Division": "SYNTHETIC",
                                "Range": "SYNTHETIC", "Beat": "SYNTHETIC"})

    manifest = {
        "dataset_name": "FireWatch synthetic forest-fire triage scenario",
        "synthetic": True,
        "real_world_observations": False,
        "real_forest_boundaries_used": False,
        "seed": int(seed),
        "observation_rows": len(observations),
        "context_rows": len(context),
        "event_count": len(scenarios),
        "event_counts": {"forest_fire": 16, "non_fire": 24},
        "class_counts": class_counts,
        "scenario_counts": {"simulated_forest_fire": 320,
                             "simulated_forest_false_alarm": 160,
                             "simulated_industrial_heat": 160,
                             "simulated_agricultural_burn": 160},
        "label_definition": "Synthetic scenario truth set by generator; not reviewed real-world ground truth.",
        "warning": "Do not use for real model accuracy, deployment decisions, or emergency response.",
    }
    return SyntheticForestDemo(_csv_bytes(OBSERVATION_FIELDS, observations),
                               _csv_bytes(CONTEXT_FIELDS, context), manifest)


def write_synthetic_forest_demo(output_dir: str | Path | None = None, *, seed: int = DEMO_SEED) -> dict:
    if output_dir is None:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        output_dir = base / "FireWatchAI-data" / "synthetic-demo"
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    demo = generate_synthetic_forest_demo(seed)
    observation_path = directory / "synthetic_firms_observations.csv"
    context_path = directory / "synthetic_forest_context.csv"
    manifest_path = directory / "synthetic_demo_manifest.json"
    observation_path.write_bytes(demo.observations_csv)
    context_path.write_bytes(demo.context_csv)
    manifest_path.write_text(json.dumps(demo.manifest, indent=2) + "\n", encoding="utf-8")
    return {**demo.manifest, "output_dir": str(directory),
            "observations_path": str(observation_path), "context_path": str(context_path),
            "manifest_path": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explicitly synthetic FireWatch forest-fire demo files.")
    parser.add_argument("--output-dir", help="Output directory (default: outside the repository in local app data)")
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    args = parser.parse_args()
    print(json.dumps(write_synthetic_forest_demo(args.output_dir, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
