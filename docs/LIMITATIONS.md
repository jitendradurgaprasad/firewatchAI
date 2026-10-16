# Limitations and future work

## Current limitations

- FIRMS detections are satellite thermal anomalies, not confirmed wildfires. No independent fire/no-fire labels, incident reports, or trusted operational outcomes are included.
- Optional FSI forest-area context comes from MODIS/VIIRS hotspots filtered using state Recorded Forest Area or Forest Cover Maps. Because FSI and FIRMS may share the underlying satellite detection, a match indicates forest-area context only and is not independent confirmation. FSI recommends combining its satellite alerts with other evidence.
- The bundled sample is a static one-day dataset (2023-07-12). Its 74,605 rows are not an independent validation set and cannot support accuracy, precision, recall, F1, probability calibration, or event generalization claims.
- Review scoring is a heuristic. Categorical confidence is mapped to ordinal constants; FRP is log scaled; weights, cutoffs and coverage are operator-configurable and not empirically calibrated. Confidence and FRP may be correlated.
- Spatial/time groups are connected components, which can form transitive chains. They are not fire perimeters, incident identifiers, or proof of common cause. Different satellite names do not necessarily imply independent confirmation.
- Association estimates use spherical haversine distances; not a high-precision local geodesic model.
- Current imported observations are held in the local SQLite database and are loaded into a Pandas dataframe for interactive analysis. CSV input is capped at 50 MiB and 100,000 rows; the live response is capped at 50 MiB. This is not a multi-gigabyte archive processor.
- The live API requires a user-supplied NASA FIRMS MAP_KEY and internet access. Credentials, quotas, network, API availability, and upstream formats can fail. The live page polls while open; this is not a durable background scheduler and it stops when the app/page is closed.
- No standalone/versioned forest-cover raster is bundled. Forest context requires an operator-downloaded FSI Fire Point Search CSV and covers only the included date/query range; unmatched records remain unknown. The FSI points are not a comprehensive land-cover mask. Jurisdiction remains unknown unless manually verified by the reviewer. OpenStreetMap base tiles provide context only and need internet access.
- General notifications are simulated. A manually triggered FormSubmit email can send a review request for a recent NASA FIRMS Area API record to the configured ERS inbox; it is not an automatic alerting service, and FormSubmit recipient activation and inbox delivery are not guaranteed. The separate SMTP path is synthetic-demo-only. There is no nearest-service lookup, automatic geographic routing, authentication/authorization, multi-user deployment, SLA, escalation, or emergency dispatch.
- The Satellite Image Viewer displays rendered GIBS true-color imagery only. It does not download full-resolution source granules, run image inference, extract hotspot fields, or connect the image to an observation. Re-fetching every 10 seconds may repeatedly display the same scene because upstream imagery publication is much slower.
- Live Monitoring polls a one-day FIRMS NRT area query once per minute while its page is open and may return unchanged results between provider updates. It visualizes FIRMS-produced detections and does not scan raw image pixels. The animated sweep is decorative. Polling requires a valid private MAP_KEY and network access.
- SQLite is local, not a shared concurrent cloud database. Reviewer IDs and contact configurations are user supplied; the app does not authenticate reviewers.
- The separate legacy image classifier is not integrated with this dashboard and is not trained by this workflow.

## Appropriate future improvements

- Collect independent field/official incident references and reviewed non-fire sources with evidence URLs and event IDs; then use event-disjoint evaluation before making any performance claim.
- Add versioned source adapters and rate-limit-aware scheduled retrieval after operational ownership and key-management policy are defined.
- Add trusted boundary/context datasets with clear licensing, date/version metadata and offline caching.
- Add role-based reviewer authentication, tamper-evident audit export and shared database only if deployment requirements justify them.
- Benchmark large regional imports and move to disk-backed spatial processing if the 50 MiB / 100,000-row interactive limit is insufficient.
- Explore clustering parameters using human-reviewed event references; keep associations separate from confirmed incident labels.
