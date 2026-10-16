"""Map helpers that only plot validated geographic coordinates."""

from __future__ import annotations

import html

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster


COLORS = {
    "HIGH REVIEW PRIORITY": "red",
    "MEDIUM REVIEW PRIORITY": "orange",
    "LOW REVIEW PRIORITY": "green",
    "INSUFFICIENT INFORMATION": "gray",
}


def map_ready_observations(observations: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return only finite in-range coordinates plus count excluded from mapping."""
    if observations.empty or not {"latitude", "longitude"}.issubset(observations.columns):
        return observations.iloc[0:0].copy(), len(observations)
    lat = pd.to_numeric(observations["latitude"], errors="coerce")
    lon = pd.to_numeric(observations["longitude"], errors="coerce")
    mask = (np.isfinite(lat) & np.isfinite(lon) & lat.between(-90, 90) & lon.between(-180, 180))
    return observations.loc[mask].copy(), int((~mask).sum())


def nearest_observation(observations: pd.DataFrame, latitude: float, longitude: float,
                        *, max_distance_km: float = 2.0) -> str | None:
    ready, _ = map_ready_observations(observations)
    if ready.empty or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    lat1, lon1 = np.radians(latitude), np.radians(longitude)
    lat2 = np.radians(ready["latitude"].astype(float).to_numpy())
    lon2 = np.radians(ready["longitude"].astype(float).to_numpy())
    delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    index = int(np.argmin(distances))
    if float(distances[index]) > max_distance_km:
        return None
    return str(ready.iloc[index]["observation_id"])


def create_observation_map(observations: pd.DataFrame, *, max_markers: int = 5000) -> tuple[folium.Map | None, int, int]:
    ready, excluded = map_ready_observations(observations)
    if ready.empty:
        return None, 0, excluded
    if "priority_score" in ready:
        ready = ready.sort_values(["priority_score", "observation_id"], ascending=[False, True])
    total = len(ready)
    if total > max_markers:
        ready = ready.head(max_markers)
    center = [float(ready["latitude"].mean()), float(ready["longitude"].mean())]
    map_widget = folium.Map(location=center, zoom_start=3, control_scale=True, tiles="OpenStreetMap")
    cluster = MarkerCluster(name="FIRMS observations", disableClusteringAtZoom=8,
                            chunkedLoading=True, showCoverageOnHover=False).add_to(map_widget)
    for row in ready.to_dict(orient="records"):
        category = str(row.get("priority_category", "INSUFFICIENT INFORMATION"))
        fields = (
            ("Observation", row.get("observation_id", "")),
            ("Coordinates", f"{float(row['latitude']):.5f}, {float(row['longitude']):.5f}"),
            ("Date/time (UTC)", row.get("acq_datetime", "")),
            ("Satellite / instrument", f"{row.get('satellite') or '—'} / {row.get('instrument') or '—'}"),
            ("Confidence", row.get("confidence") if row.get("confidence") is not None else "Missing"),
            ("FRP (MW)", row.get("frp") if row.get("frp") is not None else "Missing"),
            ("Review priority", category),
            ("Score", row.get("priority_score", "Not scored")),
            ("Explanation", row.get("explanation", "")),
            ("Review status", row.get("review_status", "Awaiting review")),
            ("Spatial/time association", row.get("association_id") or "None"),
            ("Nearby observations", row.get("nearby_observation_count", 0)),
        )
        popup_html = "<br>".join(f"<b>{html.escape(str(k))}:</b> {html.escape(str(v))}" for k, v in fields)
        folium.Marker(location=[float(row["latitude"]), float(row["longitude"])],
                      tooltip=str(row.get("observation_id", "Observation")),
                      popup=folium.Popup(popup_html, max_width=420),
                      icon=folium.Icon(color=COLORS.get(category, "gray"), icon="fire", prefix="fa")).add_to(cluster)
    folium.LayerControl().add_to(map_widget)
    return map_widget, min(total, max_markers), excluded
