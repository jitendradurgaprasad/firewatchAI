# FireWatch AI — FIRMS satellite observation triage

FireWatch AI is a local Streamlit dashboard for reviewing NASA FIRMS active-fire **thermal anomaly observations**. It ingests FIRMS-compatible CSVs, validates each source row, calculates an explainable review-priority score, displays observations on a map and table, and keeps reviewer feedback and simulated notification events in a local SQLite audit database.

A FIRMS hotspot is a satellite thermal anomaly. It is not a confirmed wildfire, a measure of fire severity, a burned-area label, or a substitute for an independent incident report. Scores only help order human review. This application does not contact emergency services or authorities.

## Run locally

Python 3.11 is recommended. From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The default installation also retains requirements for the legacy image-classification prototype and its tests. The FIRMS dashboard does not import or run the PyTorch model, require a GPU, or train a model. Set `FIREWATCH_AUTOLOAD_SAMPLE=0` to start with an empty database.

## Included sample and provenance

The first launch imports the included [NASA FIRMS Academy VIIRS SNPP sample](https://firms.modaps.eosdis.nasa.gov/content/academy/data_ingest/firms_data_ingest.html), whose acquisitions are dated 2023-07-12. Its original CSV and retrieval metadata are in `sample_data/`. The source file is 6,077,404 bytes (about 5.8 MiB), with 74,605 rows. Validation produced 74,599 unique valid rows and six exact duplicate rows. Raw confidence counts include all source rows (including duplicates); the unique valid counts are 64,895 nominal, 4,912 high, and 4,792 low. Raw product-version counts are 69,507 NRT and 5,098 URT; unique valid rows contain 69,507 NRT and 5,092 URT. All duplicates are excluded after their first occurrence and retained in the rejection audit.

NASA FIRMS describes confidence as low, nominal, or high (with product-specific encodings); it is a detection confidence field, not a probability that a wildfire exists. FIRMS active-fire observations can include non-wildfire thermal sources such as industrial heat and volcanoes. Refer to the [VIIRS 375 m product field guide](https://modaps.modaps.eosdis.nasa.gov/services/about/products/viirs-land-c2-nrt/vnp14imgtdl_nrt.html) and FIRMS documentation for interpretation.

NASA Earth science data are generally available under the [NASA Earthdata data-use policy](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy), except where specific restrictions apply. NASA requests attribution and does not endorse products using its data. Attribution is included in the sample metadata and in the app. Map tiles are provided by OpenStreetMap; follow its [copyright and attribution terms](https://www.openstreetmap.org/copyright).

The bundled sample is for software and workflow demonstration. It is one acquisition date, so it is not an independent train/validation/test dataset and cannot establish model accuracy or event generalization. This project phase performs no model training.

## Supported CSV and validation

The adapter supports comma-separated FIRMS-style data and aliases for latitude, longitude, acquisition date/time, satellite/instrument, confidence, FRP, brightness, scan/track, day/night, version, hotspot type, and jurisdiction. Latitude, longitude, acquisition date, and acquisition time are required. It accepts common FIRMS timestamp encodings and interprets acquisition time as UTC.

Every parsed input row is retained as either a valid observation or a rejected record with original source values and reason codes. It checks coordinates, dates/times, numerical fields, confidence, day/night and hotspot type. Exact duplicate source rows after the first are rejected. Missing optional values are reported and remain missing; the adapter does not infer labels or fill data. Imports are parsed in 50,000-row chunks and limited to 50 MiB and 100,000 rows for interactive use; accepted rows and rejection details are stored in SQLite.

Use **Import Data** to add an uploaded CSV or local CSV path. Provide source, product if known, and licence/attribution details. These text fields record what was supplied; the app does not independently verify user-uploaded data rights or authenticity. Review validation counts, download rejection details, then use the map/table and observation details.

### Optional NASA FIRMS live refresh

The app implements NASA's official [Area CSV API](https://firms.modaps.eosdis.nasa.gov/api/area/), which requires a free [FIRMS MAP_KEY](https://firms.modaps.eosdis.nasa.gov/api/map_key/). Put it in a private local `.env` copied from `.env.example` as `FIRMS_MAP_KEY=...`, then restart Streamlit. The app never stores or displays the key. Requests use an allowlisted NRT product, bounded west/south/east/north coordinates, 1–5 days, a timeout and a 50 MiB response ceiling. Choose **Import Data → Refresh from NASA FIRMS Area API**. Without the key or network, the bundled static sample and CSV import remain available. The live request is never described as the static sample. There is no scheduled refresh.

## Review-priority calculation

The default score combines recency (20%), FIRMS confidence (25%), FRP (25%), nearby repeat detections (25%), and input completeness (5%). FIRMS categorical confidence is mapped to fixed heuristic values (low 0.30, nominal 0.65, high 0.90); numeric confidence is interpreted on a 0–100 scale. FRP uses a configurable log scale that saturates at 500 MW. Recency uses a 72-hour half-life. Repeat detections count other observations within 1 km and 24 hours, capped at five for the factor score.

If factors are missing, remaining active weights are renormalized and missing factors are identified in the explanation. Coverage below 60%, or fewer than two available factors, yields **INSUFFICIENT INFORMATION**. Otherwise scores below 45 are low, 45–69.99 medium, and 70+ high review priority. These are configurable heuristics, not calibrated probabilities, fire severity, or operational thresholds. Confidence and FRP are related sensor signals; their combined default contribution is 50%, while custom weights may change that. Settings can change weights, thresholds, time/distance scales, FRP cap, and whether recency is referenced to the newest record or current UTC time.

Spatial/time association connects observations when both great-circle distance and UTC acquisition-time limits are met. It reports connected groups, direct-neighbor counts, time span, coordinate bounds and observed satellite names. Groups can include transitive chains and are not fire perimeters or confirmed events. Analysis runs only when requested on its page and is capped to the newest configured number of filtered observations (10,000 by default) to keep dashboard startup responsive.

## Human review, routing, and notifications

Reviewer feedback is append-only and includes outcome, assessment, optional confidence/notes/reviewer ID, and manually verified jurisdiction. Feedback is not treated as independent ground truth, and the ranking does not learn from it. Jurisdiction fields in source data are not considered verified. Active routing entries require the operator to mark the contact verified and supply a verification date; unmatched observations route to an internal review queue.

Notification creation uses an observation/trigger deduplication key and records a payload, routing reason, event history, and status. The only provider implemented and selectable by the dashboard is local simulation. No email, webhook, SMS, or authority communication is sent. Provider acceptance and delivery statuses are kept distinct in the general audit function; a simulated record is never represented as externally delivered.

## Data storage and configuration

By default, SQLite and its WAL files live outside the repository:

- Windows: `%LOCALAPPDATA%\FireWatchAI-dashboard\firewatch.sqlite3`
- Other systems: `~/.local/share/firewatch-ai/firewatch.sqlite3`

Set these optional environment variables before starting Streamlit:

- `FIREWATCH_DATA_DIR`: directory for local application state.
- `FIREWATCH_DB_PATH`: explicit SQLite path (takes precedence).
- `FIREWATCH_AUTOLOAD_SAMPLE`: `0`/`false` disables the bundled sample import; default is enabled.
- `FIREWATCH_REVIEWER_ID`: pre-fills and locks the reviewer identifier field.
- `FIREWATCH_DELIVERY_MODE`: retained for configuration clarity; the app remains simulation-only regardless of value.
- `FIRMS_MAP_KEY`: optional NASA FIRMS API key. Keep it in a private local `.env` or process environment; never commit it.

The app loads `.env` values with python-dotenv. Keep credentials and private operational data outside Git. The API key is only used for an explicit user-triggered Area API query.

## Navigation

- **Overview:** unique stored records, accepted/rejected source-row totals, review categories, feedback/notification totals, coverage and charts.
- **Import Data / Data Validation:** CSV import, row accounting, missing-field counts, and rejection exports.
- **Observation Map / Observation Table:** filtering, selectable observations, and spreadsheet-safe CSV exports.
- **Priority Review Queue / Observation Details:** sorted queue, score factors, related observations, raw record, append-only review history, notification history, and individual Markdown investigation report.
- **Spatial and Temporal Analysis:** filtered summaries of configured connected associations.
- **Feedback and Review History:** append-only audit records with outcome filtering.
- **Notification Centre:** simulation-only payloads, attempts, duplicate suppression events, retry count, and history.
- **Evaluation:** synthetic functional checks only; no accuracy, precision, recall, or real-world predictions.
- **Settings and About:** scoring weights and scales, provenance, and limitations.

CSV exports neutralize spreadsheet formula-like text values. The map requires internet access for OpenStreetMap tiles; map markers remain sourced from imported records.

## Verification

Run the suite with `python -m pytest -q`. Tests cover row accounting and duplicates, sample counts, score thresholds and factor explanations, map coordinates, spherical distance/time associations, feedback persistence and migration, API validation, notification deduplication/retry audit, CSV formula safety, investigation reports and UI empty state. Existing legacy prototype tests remain in the suite. See [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) for executed results and interpretation.

## Detailed documentation

- [Architecture](docs/ARCHITECTURE.md)
- [User guide](docs/USER_GUIDE.md)
- [Demo script and judge questions](docs/DEMO_SCRIPT.md)
- [Evaluation report](docs/EVALUATION_REPORT.md)
- [Limitations and future improvements](docs/LIMITATIONS.md)

The map needs internet access for OpenStreetMap tiles; markers remain sourced from imported records. NASA announced Suomi-NPP product delivery will cease on 2026-11-01. Live API defaults to NOAA-20 NRT and also offers NOAA-21; the bundled historical sample remains explicitly SNPP. Consult the current [FIRMS Area API product list](https://firms.modaps.eosdis.nasa.gov/api/area/) when selecting live data.

## Legacy image prototype

Older PyQt6/PyTorch image-classification, dataset-adapter, and training-guard code remains in the repository and is not used by this FIRMS dashboard. It accepts image datasets and is a separate experimental path. No classifier training was started as part of this dashboard pivot; current FIRMS thermal-anomaly rows are not image labels and must not be fed to that classifier as if they were.
