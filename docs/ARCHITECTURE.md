# Architecture

## Runtime and data flow

`app.py` is the Streamlit presentation layer. On first run it imports the bundled NASA FIRMS Academy CSV unless `FIREWATCH_AUTOLOAD_SAMPLE=0` or a local database already contains an import. Upload, local-path, and optional live Area API imports all enter the same validation and persistence path. Optional FSI forest-fire point CSVs have a separate parser/table and enrich the ranking with forest-area context.

```text
Bundled sample ─┐
CSV upload/path ├─> firms_ingest ─> validation result ─> SQLite provenance + row records
FIRMS Area API ┘                                      │
FSI point CSV ─> forest_context.py ─> local context ──┤
                                                      ├─> ranking.py ─> score explanations
                                                      ├─> association.py ─> spatial/time groups
                                                      ├─> map_view.py / Streamlit table and map
                                                      ├─> feedback + notification audit
                                                      └─> spreadsheet-safe CSV / Markdown report
```

`firewatch/firms_api.py` is optional and reads `FIRMS_MAP_KEY` from the process environment (a local `.env` is loaded if present). It performs a single bounded request, with an allowlisted product, validated bounding box, 1–5 day range, timeout, and a 50 MiB response ceiling. Errors are sanitized before display. The dashboard remains usable without internet access to the API or a key. The bundled sample and user CSV import are independent of live access.

## Ingestion and provenance

`firewatch/firms_ingest.py` accepts FIRMS-style CSV column aliases, parses rows in 50,000-row chunks, and enforces a 50 MiB and 100,000-row interactive input limit. The parser validates required coordinates and acquisition date/time, optional numeric measurements, confidence, day/night, and product type. Missing optional fields stay missing and are counted. Each row becomes either a normalized observation or a rejected row containing source-row number, original source values, and reason codes. Exact duplicate original rows after their first occurrence are rejected. The CSV bytes are bounded but buffered before chunk parsing; very large archive files are outside this interactive workflow.

`firewatch/forest_context.py` validates FSI Fire Point Search CSVs (50 MiB/100,000 rows) and matches points to FIRMS observations with a unit-sphere KD-tree, 1 km radius, and 12-hour window. It stores valid and rejected source rows separately from FIRMS observations. Unmatched points are unknown, not negative labels. FSI says its forest-fire detections are MODIS/VIIRS thermal hotspots filtered by forest-area boundaries; these can share the underlying satellite observations with FIRMS.

`firewatch/storage.py` writes imports and metadata, observations, rejected rows, forest-context imports/points, feedback, authority routing configuration, notification/event audit, evaluation runs, and settings to SQLite. The default database lives outside the repository. Source records are stored separately from normalized/derived fields and are never rewritten by feedback. Database startup uses additive schema migrations; migration does not delete feedback.

## Review score

`firewatch/ranking.py` preserves the original five-factor heuristic when no forest context is imported. With context available, the default weights are:

| Factor | Default weight | Normalization |
|---|---:|---|
| Recency | 0.17 | Exponential decay with configurable half-life |
| FIRMS confidence | 0.2125 | Low/nominal/high map to 0.30/0.65/0.90; numeric 0–100 scales linearly |
| FRP | 0.2125 | `log1p(FRP) / log1p(cap MW)`, clamped to 0–1 |
| Nearby repeat count | 0.2125 | Other records within configured haversine-equivalent radius and time window; saturates at five |
| Completeness | 0.0425 | Fraction of the four measured/review cues (recency, confidence, FRP, repeat) available |
| FSI forest-area context | 0.15 | Positive only for an FSI point within 1 km and 12 h; unknown where unmatched |

Available positive-weight factors are renormalized per observation; absent fields are omitted. Therefore no FSI import yields the same effective five-factor formula as before. An FSI match adds the forest-context weight; it is not counted as an independent label because FSI alerts are also derived from MODIS/VIIRS. The score is an investigation ordering heuristic, not probability, severity, calibrated performance, or an operational alert threshold.

## Spatial and temporal association

`firewatch/association.py` computes great-circle distances with a spherical Earth haversine formula and finds candidate neighbors using a unit-sphere KD-tree. An edge requires both distance and time-window conditions. Connected components receive deterministic IDs based on their member observation IDs and the chosen thresholds. Per-row direct neighbor counts differ from component size. Group summaries report time span, coordinate bounds, observed satellites, and maximum available FRP. A group is not asserted to be one fire. No DBSCAN is used.

## Presentation and persistence boundaries

`map_view.py` renders validated coordinates only, caps markers according to a setting, escapes popup content, and uses OpenStreetMap tiles. Base maps require internet; data analysis does not. Table/map filters operate on copies. `exports.py` neutralizes formula-like spreadsheet text and emits an individual Markdown investigation report with source record, audit context, provenance, and uncertainty disclaimer.

Feedback is append-only and can capture evidence source/reference, event ID, and an explicit independent-evidence-reviewed flag. The flag requires a decisive fire/non-fire assessment plus source, reference, and event ID; it is a reviewer attestation, not an automatic verification. Current status is the newest review record; earlier assessments remain queryable. Notification deduplication uses `SHA256(observation_id + '|' + casefold(trim(trigger_reason)))`. Duplicate attempts are persisted as `duplicate_suppressed` events on the existing notification. Only the simulation provider is enabled in the dashboard. Provider code does not create an external communication channel.

## Evaluation boundary

`firewatch/evaluation.py` tests deterministic synthetic scoring behavior. Pytest covers implementation logic and regression behavior; neither is wildfire-detection validation. There are no independent labeled fire/non-fire ground-truth records in the bundled sample.
