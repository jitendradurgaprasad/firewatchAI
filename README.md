# FireWatch AI — satellite image fire evidence

**Current phase:** satellite image fire detection (local research prototype). **Deployment is not permitted for this phase.**

FireWatch AI accepts a JPEG or PNG and is designed to return one of:

- `FIRE EVIDENCE DETECTED`
- `NO VISIBLE FIRE EVIDENCE DETECTED`
- `UNCERTAIN_OR_UNSUPPORTED`

This phase is image classification only. It does not locate or segment fire, match satellite hotspots, fuse data sources, or support emergency response. It cannot establish that every real-world fire will be detected.

## Dataset investigation and attribution

EO4WildFires was reviewed first. Its 31,730 events combine Sentinel-1/2, NASA POWER meteorology, and EFFIS wildfire annotations; example targets include `burned_mask`, and its published experiments predict burned area/severity. The dataset does not establish that visible active flames appear in a given chip, so event/burn labels are not valid active-flame labels for this classifier. Hugging Face lists CC-BY-SA-4.0 and a total size of 25.5 GB. Its loading script requires `datasets` 3.6.0 or earlier and the online viewer is disabled. It was not downloaded.

A closer candidate is the [California Sentinel-2 Active-Fire Segmentation Dataset](https://doi.org/10.5281/zenodo.22713948) (Mitra, Narimani & Farajpoor, 2026), published under CC BY 4.0. It provides 2,148 512×512 false-colour chips (R=B12/SWIR2, G=B11/SWIR1, B=B8A/NIR), masks from a modified HOTMAP SWIR threshold rule, a fire-disjoint 18/3/4 incident split, chip metadata, and independently reviewed hand masks for part of the test set. The 937.1 MB archive was not downloaded during initial inspection. The rule-derived labels are weak labels, not exhaustive human ground truth; hand-reviewed labels should be kept distinct for evaluation. The authors' code is [MIT licensed](https://github.com/MohammadrezaNarimaniUCDavis/California_Sentinel2_Active_Fire_Dataset). Cite the data as: Mitra, S., Narimani, M., & Farajpoor, P. (2026). *California Sentinel-2 Active-Fire Segmentation Dataset* (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22713948.

The corpus uses false-colour SWIR/NIR composites, not ordinary visible RGB. PNG/JPEG support here accepts only 3-channel RGB and does not imply that arbitrary RGB imagery is equivalent to the dataset's fixed-stretch spectral composite. No sample archive or model weights are currently included in this repository. No trained model or metric is claimed.

## Install

Use Python 3.11 in a virtual environment; existing system Python installations are not modified.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
```

PyTorch installation may need a platform-specific wheel; use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) if the requirements install cannot select a suitable build. CPU inference is supported. ImageNet pretrained weights are not downloaded by default.

## Local use

Check an image manifest (CSV columns: `image,label,event_id`; image paths relative to the CSV):

```powershell
python -m firewatch.cli inspect path\to\manifest.csv
```

Labels must be `ACTIVE_FIRE_EVIDENCE` or `NO_VISIBLE_FIRE_EVIDENCE`, and each row must name an event. The validator rejects missing files, invalid labels, and byte-identical duplicate files. `UNCERTAIN_OR_UNSUPPORTED` is an inference outcome, not a trainable label in this two-class pipeline. Label provenance must be established by the dataset provider; do not create labels from event association, hotspot proximity, burn scars, or absence of a satellite detection.

Train uses deterministic event-disjoint train/validation/test splits (70/15/15 by event, seed 42 by default), selects the checkpoint on validation loss, and reports metrics once on the test partition:

```powershell
python -m firewatch.cli train path\to\manifest.csv --epochs 5 --batch-size 16
```

Evaluate a saved checkpoint on the manifest's deterministic test split:

```powershell
python -m firewatch.cli evaluate path\to\manifest.csv artifacts\firewatch-resnet18.pt
```

Run inference; default abstention threshold is 0.70 and may be set with `--threshold` (valid range 0.5–1.0):

```powershell
python -m firewatch.cli predict path\to\image.png --checkpoint artifacts\firewatch-resnet18.pt --threshold 0.70
```

When weights are absent or input is unsupported, inference returns `UNCERTAIN_OR_UNSUPPORTED`; it never supplies a random or hardcoded class. The score is an uncalibrated softmax score, not a probability of real-world fire. Calibration and a separate evaluation set are needed before interpreting it probabilistically.

Supported file formats are RGB JPEG and PNG. Grayscale, alpha/multichannel PNG, TIFF, corrupted files, images smaller than 32×32, and images larger than 40 million pixels are rejected. No geographic coordinates are inferred. Preprocessing resizes to 224×224 and applies a fixed per-channel normalization; multispectral GeoTIFF support is not implemented.

## Tests

```powershell
python -m pytest
```

## Implementation status

Implemented: strict image loading, manifest and label checks, duplicate detection, event-aware splits, ResNet-18 training/checkpoint loading, basic classification metrics, structured abstaining inference, CLI, and unit tests.

Not yet implemented: dataset-specific mask-to-image-label preparation, data download/sample review, trained checkpoint, measured model evaluation, GUI, deployment, and any other FireWatch feature. Training should wait until the authorized data has been locally inspected and its weak-label limitations are reflected in the experiment.

## Development log

See [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) for phase records, test results, commits, and push status.
