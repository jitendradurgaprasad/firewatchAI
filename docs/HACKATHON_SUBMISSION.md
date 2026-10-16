# FireWatch AI — hackathon idea submission

## Project summary

**FireWatch AI is a human-review dashboard for prioritizing possible forest-fire hotspots in India.** It validates satellite thermal-anomaly records from NASA FIRMS, adds optional forest-area context from Forest Survey of India (FSI) point exports, and presents an explainable review queue with maps, source details, and an append-only feedback trail.

It is a decision-support prototype. A satellite thermal anomaly is not proof of a wildfire, and the dashboard does not dispatch alerts to emergency authorities.

## The problem

Fire-monitoring teams may need to review many satellite hotspot records with varying confidence, fire radiative power (FRP), acquisition times, and geographic context. Reviewing records without clear provenance or a consistent queue can make it harder to decide what to investigate first. Forest context can help frame a review, but it cannot independently confirm a fire when it derives from related satellite detections.

## The proposed solution

FireWatch brings these records into one local dashboard so a reviewer can:

- import and validate FIRMS-style CSV data, or optionally request a bounded recent area query from the NASA FIRMS API;
- inspect coordinates, acquisition time, sensor fields, confidence, FRP, and source provenance;
- sort observations with a configurable, factor-by-factor review-priority heuristic;
- optionally compare observations with FSI forest-fire point exports while preserving the fact that the sources can be correlated;
- inspect nearby observations, record evidence-linked human assessments, and retain an audit history;
- exercise the workflow offline using clearly marked synthetic scenarios.

The output is a **review priority**, not a calibrated probability of fire, a severity estimate, or a confirmed incident. Low information coverage can result in an abstention rather than a forced ranking.

## What is distinctive

The prototype emphasizes traceability and cautious use: rejected input rows remain inspectable; scores show their contributing factors and missing inputs; reviewer feedback is append-only; and satellite context is not mislabeled as independent ground truth. The dashboard separates processed hotspot records from a read-only rendered imagery viewer, rather than claiming it detects fires by analyzing those image pixels.

## Current workflow

1. Load the bundled historical example, import a CSV, or explicitly request a bounded FIRMS refresh.
2. Validate fields, timestamps, coordinates, and duplicates; retain source metadata and rejection reasons.
3. Review observations on the map/table and prioritize them using the displayed heuristic factors.
4. Inspect nearby records and optional FSI forest-area context.
5. Record a human assessment with source, reference, and event identifier when available.
6. Demonstrate notification handling locally in simulation; no emergency authority is contacted.

The satellite image viewer is a separate visual reference. It requests rendered NASA GIBS VIIRS imagery, but those pixels do not enter the current scoring or alert path. Refreshing every ten seconds does not mean that a new satellite capture is available every ten seconds.

## Evidence available today

- The included NASA FIRMS Academy sample contains 74,605 source rows from one acquisition date (2023-07-12). Validation found 74,599 unique valid rows and six exact duplicate rows. This is a software demonstration sample, not independently labeled wildfire ground truth.
- A deterministic synthetic workflow scenario exercises 800 generated hotspot-shaped rows across 40 generated events, plus generated forest-context points. Its labels are synthetic by construction and are not evidence of real-world accuracy.
- Automated tests and the evaluation report cover ingestion, validation, scoring logic, association rules, storage, and notification simulation. These verify software behavior, not satellite fire-detection performance.
- No model has been trained or evaluated against an independent, event-disjoint real-world fire/non-fire dataset in this dashboard phase. Precision, recall, F1, and calibrated fire probabilities are not claimed.

## Data and responsibility

The bundled sample is attributed to NASA FIRMS. NASA's data-use policy and the sample's provenance metadata are linked in the README. NASA does not endorse FireWatch. User-provided data retain their supplied source/licence notes; the application does not independently clear their rights. FSI context must be used under the source portal's terms.

This prototype is not an emergency service, operational warning system, or substitute for official incident verification. A trained reviewer should corroborate any potential event with independent evidence before action.

## Next development steps

1. Collect independently sourced incident confirmations and reviewed non-fire cases, with evidence references and stable event IDs.
2. Establish event/source-disjoint training, validation, and test partitions before reporting any model performance.
3. Evaluate coverage, false-alarm burden, missed events, calibration, and performance across seasons, sensors, and regions with domain review.
4. Only after those checks, assess whether a satellite-image model or an operational deployment is warranted.

## Short pitch

> FireWatch AI helps reviewers organize and investigate possible forest-fire hotspots from satellite thermal-anomaly feeds. It validates data, explains why a record is prioritized, adds carefully labeled forest-area context, and preserves evidence-linked human review. It is designed as a transparent triage prototype: it does not claim to confirm fires or replace independent verification.

## Demo guide

Use the five-minute walkthrough in [DEMO_SCRIPT.md](DEMO_SCRIPT.md). For a reliable offline demonstration, load the generated synthetic scenario and label every screen and result as synthetic. For the FIRMS sample, explain that the records are historical hotspot observations and do not have independent incident labels.
