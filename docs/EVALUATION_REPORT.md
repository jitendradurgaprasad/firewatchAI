# Evaluation report — FireWatch AI

Evaluation date: 2026-10-16. This report describes the bundled data and checks actually executed during this development run. Functional software checks are not satellite fire-detection performance.

## A. Dataset integrity and provenance

- Source: [NASA FIRMS Academy data-ingestion tutorial and VIIRS SNPP sample](https://firms.modaps.eosdis.nasa.gov/content/academy/data_ingest/firms_data_ingest.html).
- Local file: `sample_data/nasa_firms_viirs_snpp_071223.csv`; provenance is in its adjacent JSON file.
- File size: 6,077,404 bytes. SHA-256: `edf79f62b35ba4485d7c85e01be5f6f61d30ab7b4addcf2eea00e2581982c30f`. Streamlit verifies size and SHA-256 against the manifest before bootstrapping the sample.
- Source CSV: 74,605 rows, 14 columns, one acquisition date (2023-07-12), UTC acquisition range 00:03–19:50.
- Raw confidence rows before duplicate removal: nominal `n` 64,901; high `h` 4,912; low `l` 4,792.
- Raw product versions: `2.0NRT` 69,507; `2.0URT` 5,098.
- Attribution is recorded in the sample manifest and app. NASA ESDIS requests source acknowledgement and does not endorse products using NASA data. See [NASA Earthdata data-use policy](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy).

## B–E. Ingestion, validation, duplicates and coordinates

Executed adapter result against the bundled file: 74,605 total rows, 74,599 valid unique observations, and six rejected rows. All six rejections were exact duplicate records; no other malformed coordinates, timestamps, confidence, or numeric measurements were found. Rejected source values and reasons remain available. Accepted rows have in-range coordinates and valid UTC timestamps.

After duplicate removal, unique valid confidence counts are nominal 64,895, high 4,912, low 4,792. Unique product-version counts are NRT 69,507 and URT 5,092. Raw metadata counts are not added to accepted counts; accepted plus rejected equals the 74,605 source rows.

Validation tests also covered malformed/unsupported schemas, empty input, blank rows, aliases, invalid coordinate/date/time/confidence/FRP values, exact duplicates, input size and row limits. These are controlled adapter tests, not additional observed dataset rows.

## F–G. Priority calculations and explanations

The existing five-factor formula was retained and is documented in [ARCHITECTURE.md](ARCHITECTURE.md): recency 0.20, confidence 0.25, FRP 0.25, nearby repeat 0.25, completeness 0.05. Available positive-weight factors renormalize per observation; missing factors are explicit and insufficient coverage abstains. The thresholds default to 45 (medium) and 70 (high), and can be configured.

The functional evaluation uses six explicitly synthetic records and passed 11/11 checks: deterministic scores, signal ordering, missing-input explanation and abstention, bounded values, custom weights, default weight sum, elapsed-hour repeat gating, space/time association, distance units, and synthetic group summary. Separate tests exercise the 44.99 / 45 / 69.99 / 70 category boundaries and factor-contribution consistency. This is not real-world accuracy evaluation.

A regression check normalizes Pandas timestamps to nanoseconds before integer comparison. This environment stores UTC timestamps at microsecond precision; repeat detection and association now compare actual elapsed time.

## H. Spatial and temporal association

Synthetic tests verified that a nearby pair inside the configured time window associates and a spatially close record outside that time window does not. Haversine distance from (0°, 0°) to (0°, 1°) is approximately 111.195 km.

A direct module benchmark on the first 10,000 sample rows completed in 12.48 seconds, with 1,381 groups and 9,078 associated observations at 5 km / 6 h. The sample-backed Streamlit interaction analyzed the newest 10,000 of 74,599 unique observations and reported 1,161 groups and 8,945 associated observations; 1,055 analyzed rows had no qualifying pair and 64,599 were outside the analysis cap. The dashboard reports this limit. These machine-specific results do not imply common fire incidents: groups can be transitive and are not perimeters.

## I–J. Feedback and notifications

SQLite tests verified append-only feedback, latest-review status derivation, internal-queue fallback for unknown jurisdiction, additive migration preserving an existing feedback row, and provenance links when the same observation is imported from multiple sources.

Notification tests verified payload disclaimers, simulation status, deduplication, persisted `duplicate_suppressed` audit event/count, and retry status history. No email, SMS, webhook or authority message was sent. A live NASA API key was not configured; API response handling was tested using a mocked response.

## K. Exports

Tests verified formula-like spreadsheet values are neutralized in CSV and the individual Markdown report contains source attribution, original source row and an unverified-anomaly disclaimer. Page CSV exports use the active filter or the documented analysis subset.

## L. Application startup

Streamlit AppTest bootstrapped the bundled sample, displayed 74,599 valid observations and six rejections, rendered the Overview with no app exceptions, navigated to Spatial and Temporal Analysis, calculated the configured subset, and rendered association metrics with no app exceptions. An empty-database AppTest exercised every navigation page, and a controlled synthetic populated AppTest verified the spatial page’s associated/group/unpaired/not-analyzed metrics without app exceptions. Full-sample startup is CPU-based and took approximately 47 seconds in the recorded local Overview smoke run; spatial association is user-triggered.

## M. Automated checks

Final verification commands:

```text
python -m pytest -q -rA                         42 passed
python -m compileall -q app.py firewatch tests  passed
git diff --check                               passed
```

The 42 tests include the retained legacy image-prototype suite. They do not imply that the old prototype is integrated with this FIRMS dashboard.

## N. Forest-fire triage extension (2026-10-16)

The project scope is now India-focused forest-fire hotspot triage. Optional CSVs from the [FSI Fire Point Search](https://fsiforestfire.gov.in/FirePointSearch/Search) are parsed into a separate local context table; test fixtures use the portal's published Fire Date / Time / Source / Lat / Lon fields. FSI defines a forest-fire detection as a MODIS/VIIRS hotspot filtered to Recorded Forest Area or Forest Cover Map boundaries ([FSI FAQ](https://fsiforestfire.gov.in/document/faq.pdf)). Since those detections may be the same satellite observations as FIRMS, this is a forest-area context feature, never ground truth.

The new context factor is 15% when a point matches within 1 km and 12 hours. Unmatched context stays unknown. When no FSI context is imported, the existing five-factor score is unchanged after weight normalization. Human review now records an optional evidence source/reference, event ID, and an explicit independent-evidence-reviewed attestation; the attestation is gated on a decisive fire/non-fire outcome and all three fields. These fields prepare an event-disjoint evaluation dataset but do not claim accuracy.

Using the configured local NASA key, one current NOAA-20 NRT query for the dashboard's default India-area box returned 289 rows; FireWatch accepted all 289. The CSV was saved outside the repository under `%LOCALAPPDATA%\FireWatchAI-data\FIRMS\live\`. This validates live acquisition and parsing only. It contains no independent fire labels. No FSI export was ingested, no ground-truth labels were created, and no model was trained.

## Accuracy boundary

The bundled sample contains thermal anomalies and no independent incident-level wildfire truth labels. Precision, recall, F1, probability calibration, and wildfire-model accuracy are therefore not meaningful claims for this application. Synthetic checks verify software behavior only. See [LIMITATIONS.md](LIMITATIONS.md) for known constraints.

## Synthetic end-to-end scenario

Added deterministic generator `python -m firewatch.synthetic_demo` and an Import Data button that generates and loads the same scenario. Verification in this run: 800/800 observation rows accepted (320 generator-labeled forest-fire, 480 generator-labeled non-fire), 480/480 context rows accepted, 40 generated events, and context matching preserves the explicit synthetic source label. Files and manifest were written outside the repository under `%LOCALAPPDATA%\\FireWatchAI-data\\synthetic-demo`. Labels are generator-defined by construction; this confirms the pipeline can process its own scenario, not that it detects real fires. No training or accuracy measurement was performed.
