# Development log

## Phase A — repository inspection

- Inspected an empty checkout on `main`; no source files, commits, configured remotes, dependencies, tests, datasets, model weights, or uncommitted user changes were present.
- Python 3.11.3 and 3.14 are installed. Pillow, NumPy, PyTorch, torchvision, pytest, and scikit-learn are importable under Python 3.11; rasterio and Hugging Face `datasets` are absent. CUDA availability has not been needed yet.
- Target GitHub URL responded to `git ls-remote` with no refs. Existing repository remote is absent. Initial push will create the first remote branch if authentication permits.
- Dataset research: EO4WildFires (25.5 GB, CC-BY-SA-4.0) labels wildfire/burned-area outcomes rather than visible flames; not downloaded. Identified the California Sentinel-2 Active-Fire Segmentation Dataset (937.1 MB, CC BY 4.0) with weak SWIR-rule labels, hand-reviewed subset, and event split. Download requested separately; no archive has been fetched yet.
- No changes were made during inspection.

## Phase B — image and ML foundation (in progress)

- Implemented strict JPEG/PNG image validation, manifest validation, content-hash duplicate detection, event-disjoint splitting, ResNet-18 training and checkpoint metadata, structured abstaining inference, metrics, CLI, tests, and initial documentation.
- Training/inference have not been run; no weights or actual labeled samples are available yet. Tests and Git gate pending.
- Files: `firewatch/`, `tests/`, `requirements.txt`, `pytest.ini`, `.gitignore`, `README.md`, `DEVELOPMENT_LOG.md`.
- Commit/push: pending.
