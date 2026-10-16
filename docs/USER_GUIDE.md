# User guide

## 1. Install and start

Use Python 3.11 and run these commands in PowerShell from the repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first launch loads the bundled public Academy sample into a SQLite database under `%LOCALAPPDATA%\FireWatchAI-dashboard`. The sample is static (retrieved 2026-10-16, acquisition date 2023-07-12), not live. Set `FIREWATCH_AUTOLOAD_SAMPLE=0` for an empty dashboard. To preserve a chosen database, set `FIREWATCH_DATA_DIR` or `FIREWATCH_DB_PATH` before launch.

## 2. Import observations

Open **Import Data**. Either select a CSV or enter a local file path; do not provide both. Add the source description, product if known, and attribution/licence note. Uploads and paths are limited to 50 MiB and 100,000 rows. Review the reported row counts, source date coverage, available fields, missing fields, and rejected-row reasons on **Data Validation**. The validation/rejection CSV exports retain original fields. FIRMS acquisition times are normalized to UTC. Duplicate rows after the first copy are rejected.

### Add forest-area context from FSI

On the FSI [Forest Fire Point Search](https://fsiforestfire.gov.in/FirePointSearch/Search), filter the dates and area you need and use **Download CSV**. In FireWatch's **Import Data** page, upload that export under **Import Forest Survey of India forest-area context**. FireWatch validates the date/time, sensor, coordinates, and duplicate points while retaining source rows. It matches within 1 km and 12 hours; unmatched observations remain unknown and are not penalized.

FSI defines forest-fire detections as MODIS/VIIRS thermal hotspots filtered by forest-area boundaries (Recorded Forest Area or Forest Cover Maps). FSI and FIRMS can share the same satellite hotspot, so a match adds forest-area context but is not independent confirmation. [FSI FAQ](https://fsiforestfire.gov.in/document/faq.pdf). Check the FSI portal's terms before redistributing its exports; FireWatch stores them locally.

### Optional recent FIRMS refresh

The official [FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/) requires a free MAP_KEY. Request one from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/map_key/), then put it in a local `.env` file copied from `.env.example`:

```text
FIRMS_MAP_KEY=your-private-key
```

Keep `.env` private. Restart Streamlit, visit **Import Data**, select an allowlisted NRT product, set `west,south,east,north` and 1–5 recent days, then choose **Fetch and validate recent FIRMS CSV**. A regional area is recommended because a world query can return many tens of thousands of rows per day and can use multiple transactions. API errors leave existing data intact. Never post or commit the key. If no key/network is available, use the bundled data or upload a downloaded CSV.

## 3. Explore

- **Overview** summarizes unique stored observations, source import row counts, rejects, priority categories, waiting reviews, feedback, and notifications. Reimporting existing observations can mean imported accepted rows exceed unique stored observations.
- **Observation Map** filters by date, satellite, confidence, priority, status, search, FRP and optional bounding box. Marker colors show review priority. Click a marker and open **Observation Details** to inspect it. Tiles require internet and are attributed to OpenStreetMap.
- **Observation Table** has the same filters, text search, row selection, 25–250 row pagination, and filtered CSV export. Sort columns in the table. Selecting a row sets the observation for details.
- **Priority Review Queue** filters to awaiting-review items and orders them by the current score.
- **Spatial and Temporal Analysis** reports connected groups under the configured distance/time rule. A group may be transitive; it is not an incident assertion. Satellite counts only summarize source fields.

## 4. Review one observation

Open **Observation Details**, select the record, and inspect source/product, UTC time, coordinates, sensor fields, score and factor explanation, missing data, association, raw source row, previous feedback, and notification history. The Markdown investigation report contains this information and an unverified-anomaly disclaimer.

Use **Record a human review** to append an outcome and optional assessment, confidence, notes, reviewer ID, event ID, and evidence source/reference. Mark evidence independent only when it is not derived from the FIRMS/VIIRS hotspot. A decisive fire/non-fire assessment can be marked for future evaluation only when a source, reference, and event ID are supplied. Each submission creates a new row; it does not change measurements or erase previous reviews.

## 5. Simulated notifications

In **Notification Centre**, select an observation and provide a trigger purpose. Unknown jurisdiction routes to the internal review queue. The app creates a deduplicated payload and records local simulation status; no email, webhook, SMS, or authority message is sent. Repeat attempts with the same observation and purpose are suppressed and added to notification audit history.

## 6. Configure and export

**Settings and About** lets you set six weights, score thresholds, coverage minimum, recency half-life, repeat radius/window, FRP saturation scale, association enablement/distance/time window, and map cap. Weights must be non-negative with positive sum; medium threshold must remain below high. Settings persist in SQLite. Without FSI data, the missing forest-context factor is omitted and the previous five-factor ranking is preserved after normalization.

Use page-level CSV downloads for filtered observations, queue, association summary, rejected rows, validation summary, feedback, notifications, and evaluation history. Individual reports download as Markdown. CSV string fields are protected against common spreadsheet formula injection. Exports do not contain FIRMS API credentials.

## 7. Run checks

```powershell
python -m pytest -q
python -m compileall -q app.py firewatch tests
```

Evaluation page checks are synthetic functional checks, not fire-detection accuracy metrics.
