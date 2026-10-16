# Limitations and future work

## Current limitations

- FIRMS detections are satellite thermal anomalies, not confirmed wildfires. No independent fire/no-fire labels, incident reports, or trusted operational outcomes are included.
- The bundled sample is a static one-day dataset (2023-07-12). Its 74,605 rows are not an independent validation set and cannot support accuracy, precision, recall, F1, probability calibration, or event generalization claims.
- Review scoring is a heuristic. Categorical confidence is mapped to ordinal constants; FRP is log scaled; weights, cutoffs and coverage are operator-configurable and not empirically calibrated. Confidence and FRP may be correlated.
- Spatial/time groups are connected components, which can form transitive chains. They are not fire perimeters, incident identifiers, or proof of common cause. Different satellite names do not necessarily imply independent confirmation.
- Association estimates use spherical haversine distances; not a high-precision local geodesic model.
- Current imported observations are held in the local SQLite database and are loaded into a Pandas dataframe for interactive analysis. CSV input is capped at 50 MiB and 100,000 rows; the live response is capped at 50 MiB. This is not a multi-gigabyte archive processor.
- The live API requires a user-supplied NASA FIRMS MAP_KEY and internet access. Credentials, quotas, network, API availability, and upstream formats can fail. No background scheduler or automatic refresh exists.
- No administrative-boundary reverse geocoder, protected-area, settlement, road, or land-cover layer is included. Jurisdiction remains unknown unless manually verified by the reviewer. OpenStreetMap base tiles provide context only and need internet access.
- Notifications are simulation-only. There is no operational messaging, authentication/authorization, multi-user deployment, SLA, alert escalation, or emergency dispatch.
- SQLite is local, not a shared concurrent cloud database. Reviewer IDs and contact configurations are user supplied; the app does not authenticate reviewers.
- The separate legacy image classifier is not integrated with this dashboard and is not trained by this workflow.

## Appropriate future improvements

- Add independent incident reference data and a pre-registered, event-disjoint evaluation protocol before making any model-performance claim.
- Add versioned source adapters and rate-limit-aware scheduled retrieval after operational ownership and key-management policy are defined.
- Add trusted boundary/context datasets with clear licensing, date/version metadata and offline caching.
- Add role-based reviewer authentication, tamper-evident audit export and shared database only if deployment requirements justify them.
- Benchmark large regional imports and move to disk-backed spatial processing if the 50 MiB / 100,000-row interactive limit is insufficient.
- Explore clustering parameters using human-reviewed event references; keep associations separate from confirmed incident labels.
