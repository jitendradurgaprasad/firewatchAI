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

The first launch imports the included [NASA FIRMS Academy VIIRS SNPP sample](https://firms.modaps.eosdis.nasa.gov/content/academy/data_ingest/firms_data_ingest.html), whose acquisitions are dated 2023-07-12. Its original CSV and retrieval metadata are in `sample_data/`. The source file is 6,077,404 bytes (about 5.8 MiB), with 74,605 rows. Validation produced 74,599 valid rows and six exact duplicate rows. All six duplicates are excluded after their first occurrence and retained in the rejected-row audit.

NASA FIRMS describes confidence as low, nominal, or high (with product-specific encodings); it is a detection confidence field, not a probability that a wildfire exists. FIRMS active-fire observations can include non-wildfire thermal sources such as industrial heat and volcanoes. Refer to the [VIIRS 375 m product field guide](https://modaps.modaps.eosdis.nasa.gov/services/about/products/viirs-land-c2-nrt/vnp14imgtdl_nrt.html) and FIRMS documentation for interpretation.

NASA Earth science data are generally available under the [NASA Earthdata data-use policy](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy), except where specific restrictions apply. NASA requests attribution and does not endorse products using its data. Attribution is included in the sample metadata and in the app. Map tiles are provided by OpenStreetMap; follow its [copyright and attribution terms](https://www.openstreetmap.org/copyright).

The bundled sample is for software and workflow demonstration. It is one acquisition date, so it is not an independent train/validation/test dataset and cannot establish model accuracy or event generalization. This project phase performs no model training.

## Supported CSV and validation

The adapter supports comma-separated FIRMS-style data and aliases for latitude, longitude, acquisition date/time, satellite/instrument, confidence, FRP, brightness, scan/track, day/night, version, hotspot type, and jurisdiction. Latitude, longitude, acquisition date, and acquisition time are required. It accepts common FIRMS timestamp encodings and interprets acquisition time as UTC.

Every parsed input row is retained as either a valid observation or a rejected record with original source values and reason codes. It checks coordinates, dates/times, numerical fields, confidence, day/night and hotspot type. Exact duplicate source rows after the first are rejected. Missing optional values are reported and remain missing; the adapter does not infer labels or fill data. Imports are chunk-read to bound CSV parsing memory, while valid and rejected rows are stored in SQLite.

Use **Import Data** to add a CSV. The uploader must provide source and licence/attribution details. These text fields record what was supplied; the app does not independently verify user-uploaded data rights or authenticity. Review validation counts, download rejection details, then use the map/table and observation details.

## Review-priority calculation

The default score combines recency (20%), FIRMS confidence (25%), FRP (25%), nearby repeat detections (25%), and input completeness (5%). FIRMS categorical confidence is mapped to fixed heuristic values (low 0.30, nominal 0.65, high 0.90); numeric confidence is interpreted on a 0–100 scale. FRP uses a configurable log scale that saturates at 500 MW. Recency uses a 72-hour half-life. Repeat detections count other observations within 1 km and 24 hours, capped at five for the factor score.

If factors are missing, remaining active weights are renormalized and missing factors are identified in the explanation. Coverage below 60%, or fewer than two available factors, yields **INSUFFICIENT INFORMATION**. Otherwise scores below 45 are low, 45–69.99 medium, and 70+ high review priority. These are configurable heuristics, not calibrated probabilities, fire severity, or operational thresholds. Confidence and FRP are related sensor signals; their combined default contribution is capped at 50%. Settings can change weights, time/distance scales, FRP cap, and whether recency is referenced to the newest record or current UTC time.

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
- `FIREWATCH_DELIVERY_MODE`: documented deployment intent; dashboard currently remains simulation-only regardless of value.

Copy `.env.example` as a starting point if your launch environment loads dotenv files. Keep credentials and private operational data outside Git. There is no FIRMS API key configured and live data fetching is not implemented; users can import a downloaded CSV.

## Navigation

- **Overview:** counts, queue summary, charts, and next review items.
- **Import Data / Data Validation:** CSV import, row accounting, missing-field counts, and rejection exports.
- **Observation Map / Observation Table:** filtering, selectable observations, and spreadsheet-safe CSV exports.
- **Priority Review Queue / Observation Details:** sorted work queue, score explanation, original record, and human review form.
- **Feedback and Review History:** append-only audit records.
- **Notification Centre:** simulation-only notification records and history.
- **Evaluation:** synthetic functional checks only; no accuracy, precision, recall, or real-world predictions.
- **Settings and About:** scoring weights and scales, provenance, and limitations.

CSV exports neutralize spreadsheet formula-like text values. The map requires internet access for OpenStreetMap tiles; map markers remain sourced from imported records.

## Verification

Run the suite with `python -m pytest -q`. Dashboard tests cover row accounting and duplicates, the sample’s observed counts, score behavior, map coordinate checks, feedback persistence, routing, notification deduplication/retry audit, CSV formula safety, and functional scoring checks. Existing legacy prototype tests remain in the suite.

## Legacy image prototype

Older PyQt6/PyTorch image-classification, dataset-adapter, and training-guard code remains in the repository and is not used by this FIRMS dashboard. It accepts image datasets and is a separate experimental path. No classifier training was started as part of this dashboard pivot; current FIRMS thermal-anomaly rows are not image labels and must not be fed to that classifier as if they were.
