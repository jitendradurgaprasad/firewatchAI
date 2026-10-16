# Hackathon demonstration script

## Three-minute version

1. **0:00–0:25 — Overview.** Launch with `streamlit run app.py`. Point out that the app reviews satellite thermal anomalies and never calls them confirmed wildfires. Show imported/accepted/rejected counts and one-day sample provenance.
2. **0:25–0:55 — Provenance and validation.** Open **Data Validation**. Show 74,605 parsed rows, 74,599 accepted and six exact duplicates rejected. Download the validation report or rejected rows.
3. **0:55–1:25 — Map and filters.** Open **Observation Map**. Filter confidence/date or a region; explain marker color is review priority. Click a point, then open **Observation Details**.
4. **1:25–2:05 — Explainable review.** Show score, normalized factors, missing features and association explanation. Open the original source record. State that this score orders review and is not a fire probability or severity estimate.
5. **2:05–2:35 — Human audit.** Submit a clearly labeled review. Show it appears in feedback history and does not overwrite the observation.
6. **2:35–3:00 — Safe follow-up.** Create a notification simulation. Repeat the same trigger to show dedupe suppression. Show “simulation only” and that no message was sent.

Expected sample facts: static NASA FIRMS Academy VIIRS SNPP sample, acquisition date 2023-07-12, 74,605 source rows, 74,599 valid, six duplicate rejections; 64,901 nominal, 4,912 high, 4,792 low; versions 69,507 NRT and 5,098 URT. These are verified from the bundled CSV during evaluation, not hard-coded into the application.

## Five-minute version

1. **0:00–0:35 — Mission and Overview.** Explain thermal anomalies, uncertainty, review counts, source coverage and no dispatch.
2. **0:35–1:15 — Validation.** Show the import fields, accepted/rejected accounting, duplicate rule, optional field counts, and source metadata.
3. **1:15–2:00 — Explorer.** Demonstrate map filters, bounding box, FRP, date, status, table search and pagination. Click a point to select it for details.
4. **2:00–2:45 — Score.** Explain recency exponential decay, confidence ordinal mapping, log FRP scaling, repeat count, completeness, missing-weight renormalization, coverage abstention and configurable thresholds.
5. **2:45–3:25 — Spatial/time groups.** Show distance/time connected components and group extent/satellite count. Explain that transitive proximity does not establish one fire and source satellite counts are not independent ground truth.
6. **3:25–4:10 — Review.** Show source fields, association, prior review and individual Markdown report. Add a human assessment; show append-only history.
7. **4:10–4:40 — Notification.** Show the local simulated payload, unknown jurisdiction wording, duplicate attempt audit and no external send.
8. **4:40–5:00 — Extensibility and limits.** Explain optional NASA Area API requires a local free MAP_KEY, is capped to a selected area/time range, and failure leaves sample/upload available. Mention no ground-truth accuracy claim.

## Optional live-data demonstration

Only attempt if the MAP_KEY has already been configured privately. **Import Data → Refresh from NASA FIRMS Area API**, choose `VIIRS_NOAA20_NRT`, retain a bounded South Asia extent (`60,5,100,38`) and one day, and fetch. Show retrieval timestamp and row validation. If credentials/network fail, report that honestly and continue with the static sample; never call it live data.

## Likely judge questions

**Is a marker a confirmed wildfire?** No. FIRMS indicates a satellite thermal anomaly. It can be a non-fire heat source; a person must investigate.

**Is 81% an 81% chance of fire?** No. It is a configurable weighted review-priority score, not a calibrated probability.

**How do you know it works?** We validate schema, row accounting, timestamps, coordinates, formula consistency, association distance/time logic, audit persistence, deduplication and exports with automated tests and an interactive startup smoke check. We do not claim fire-detection accuracy because this sample has no independent incident labels.

**Does it notify emergency services?** No. It stores a local simulation event only.

**What if the NASA API key or network is unavailable?** Use the included static sample or import a local CSV. The API is optional and errors are shown without replacing data.

**Are clusters fires?** No. They are graph-connected observations under configured distance and time limits, for review organization.
