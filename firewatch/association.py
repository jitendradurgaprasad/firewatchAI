"""Distance and time-window association for satellite thermal anomalies.

Clusters are connected components in a graph whose edges satisfy both a
haversine distance and time-window condition. Components are review aids, not
fire perimeters or proof of a common incident.
"""
from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088


def haversine_km(latitude_a: float, longitude_a: float,
                 latitude_b: float, longitude_b: float) -> float:
    values = (latitude_a, longitude_a, latitude_b, longitude_b)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Coordinates must be finite numbers.")
    if not (-90 <= latitude_a <= 90 and -90 <= latitude_b <= 90 and
            -180 <= longitude_a <= 180 and -180 <= longitude_b <= 180):
        raise ValueError("Coordinates must be inside geographic bounds.")
    lat1, lat2 = math.radians(latitude_a), math.radians(latitude_b)
    dlat, dlon = lat2 - lat1, math.radians(longitude_b - longitude_a)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def associate_observations(observations: pd.DataFrame, *, distance_km: float = 5.0,
                           time_window_hours: float = 6.0,
                           include_singletons: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add deterministic association IDs, neighbor counts and summary table.

    Missing/invalid timestamps and invalid coordinates are excluded from
    association, but rows are retained with null association and a reason.
    The IDs are stable for identical input IDs and parameters.
    """
    if not math.isfinite(distance_km) or distance_km <= 0 or distance_km > 500:
        raise ValueError("distance_km must be between 0 and 500.")
    if not math.isfinite(time_window_hours) or time_window_hours <= 0 or time_window_hours > 720:
        raise ValueError("time_window_hours must be between 0 and 720.")
    result = observations.copy()
    if result.empty:
        for key in ("association_id", "nearby_observation_count", "association_status"):
            result[key] = pd.Series(dtype="object")
        return result, pd.DataFrame()
    required = {"observation_id", "latitude", "longitude", "acq_datetime"}
    if not required.issubset(result.columns):
        raise ValueError(f"Observations must include {', '.join(sorted(required))}.")
    lat = pd.to_numeric(result.latitude, errors="coerce")
    lon = pd.to_numeric(result.longitude, errors="coerce")
    dates = pd.to_datetime(result.acq_datetime, utc=True, errors="coerce")
    eligible_mask = (np.isfinite(lat) & np.isfinite(lon) & lat.between(-90, 90) &
                     lon.between(-180, 180) & dates.notna())
    valid_indices = np.flatnonzero(eligible_mask.to_numpy())
    ids = result.observation_id.astype(str).tolist()
    parent = list(range(len(result)))
    degree = [0] * len(result)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    if len(valid_indices) > 1:
        coords = np.radians(np.column_stack((lat.iloc[valid_indices], lon.iloc[valid_indices])))
        xyz = np.column_stack((np.cos(coords[:, 0]) * np.cos(coords[:, 1]),
                               np.cos(coords[:, 0]) * np.sin(coords[:, 1]), np.sin(coords[:, 0])))
        chord = 2 * math.sin(distance_km / (2 * EARTH_RADIUS_KM))
        times = dates.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        window_ns = int(time_window_hours * 3600 * 1_000_000_000)
        tree = cKDTree(xyz)
        # Query in bounded batches so dense regional hotspots do not create a
        # second full-size list-of-lists at once. Time filtering is vectorized
        # before the union-find loop, which avoids Python work for spatially
        # close but temporally unrelated observations.
        batch_size = 2048
        for start in range(0, len(valid_indices), batch_size):
            stop = min(start + batch_size, len(valid_indices))
            neighbors = tree.query_ball_point(xyz[start:stop], chord)
            for offset, local_neighbors in enumerate(neighbors):
                local_i = start + offset
                candidates = np.asarray(local_neighbors, dtype=np.int64)
                candidates = candidates[candidates > local_i]
                if not len(candidates):
                    continue
                candidates = candidates[np.abs(times[valid_indices[candidates]] - times[valid_indices[local_i]]) <= window_ns]
                for local_j in candidates:
                    i, j = int(valid_indices[local_i]), int(valid_indices[local_j])
                    degree[i] += 1
                    degree[j] += 1
                    union(i, j)

    components: dict[int, list[int]] = {}
    for idx in valid_indices:
        root = find(int(idx))
        if degree[idx] > 0 or include_singletons:
            components.setdefault(root, []).append(int(idx))
    association_ids: list[str | None] = [None] * len(result)
    counts = [0] * len(result)
    statuses = ["eligible_no_nearby_pair" if eligible_mask.iloc[i] else "missing_or_invalid_coordinate_or_time"
                for i in range(len(result))]
    summaries = []
    for members in components.values():
        component_ids = sorted(ids[i] for i in members)
        basis = "|".join(component_ids) + f"|{distance_km:g}|{time_window_hours:g}"
        assoc_id = "assoc_" + uuid.uuid5(uuid.NAMESPACE_URL, basis).hex[:16]
        group_times = dates.iloc[members]
        group_lats, group_lons = lat.iloc[members], lon.iloc[members]
        satellites = sorted({str(result.iloc[i].get("satellite") or "Unknown") for i in members})
        frp = pd.to_numeric(result.iloc[members].get("frp", pd.Series(dtype=float)), errors="coerce")
        for i in members:
            association_ids[i] = assoc_id
            counts[i] = degree[i]
            statuses[i] = "associated"
        summaries.append({"association_id": assoc_id, "observation_count": len(members),
                          "time_start_utc": group_times.min().isoformat(),
                          "time_end_utc": group_times.max().isoformat(),
                          "latitude_min": float(group_lats.min()), "latitude_max": float(group_lats.max()),
                          "longitude_min": float(group_lons.min()), "longitude_max": float(group_lons.max()),
                          "satellites": ", ".join(satellites), "distinct_satellite_count": len(satellites),
                          "max_frp_mw": float(frp.max()) if frp.notna().any() else None,
                          "association_rule": f"Connected components, ≤{distance_km:g} km and ≤{time_window_hours:g} h edges"})
    result["association_id"] = association_ids
    result["nearby_observation_count"] = counts
    result["association_status"] = statuses
    summary = pd.DataFrame(summaries).sort_values("observation_count", ascending=False) if summaries else pd.DataFrame()
    return result, summary
