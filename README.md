# FireWatch AI — satellite image fire evidence

**Current phase:** satellite image fire detection (local research prototype). **Deployment is not permitted for this phase.**

FireWatch AI accepts a JPEG or PNG and is designed to return one of:

- `FIRE EVIDENCE DETECTED`
- `NO VISIBLE FIRE EVIDENCE DETECTED`
- `UNCERTAIN_OR_UNSUPPORTED`

This phase is image classification only. It does not locate or segment fire, match satellite hotspots, fuse data sources, or support emergency response. It cannot establish that every real-world fire will be detected.

## Dataset investigation and attribution

EO4WildFires was reviewed first. Its 31,730 events combine Sentinel-1/2, NASA POWER meteorology, and EFFIS wildfire annotations; example targets include `burned_mask`, and its published experiments predict burned area/severity. The dataset does not establish that visible active flames appear in a given chip, so event/burn labels are not valid active-flame labels for this classifier. Hugging Face lists CC-BY-SA-4.0 and a total size of 25.5 GB. Its loading script requires `datasets` 3.6.0 or earlier and the online viewer is disabled. It was not downloaded.

A closer candidate is the [California Sentinel-2 Active-Fire Segmentation Dataset](https://doi.org/10.5281/zenodo.22713948) (Mitra, Narimani & Farajpoor, 2026), published under CC BY 4.0. Its 937,057,654-byte archive was downloaded for sample inspection and verified against the published MD5; it is stored outside this repository. The archive contains 2,148 512×512 false-colour chips (R=B12/SWIR2, G=B11/SWIR1, B=B8A/NIR), masks from a modified HOTMAP SWIR threshold rule, a fire-disjoint 18/3/4 incident split, and chip metadata. The splits contain 1,470/246/432 chips and 548/137/156 mask-positive chips for train/validation/test. The rule-derived labels are weak labels, not exhaustive human ground truth. Mask value 255 is no-data, not negative. There are 233 independently reviewed test mask PNGs, but the Label Studio JSON needed to identify reviewed empty and uncertain chips is absent from the archive, so this subset is not currently used as a complete independent image-classification evaluation set. The authors' code is [MIT licensed](https://github.com/MohammadrezaNarimaniUCDavis/California_Sentinel2_Active_Fire_Dataset). Cite the data as: Mitra, S., Narimani, M., & Farajpoor, P. (2026). *California Sentinel-2 Active-Fire Segmentation Dataset* (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22713948.

The corpus uses false-colour SWIR/NIR composites, not ordinary visible RGB. PNG/JPEG support here accepts only 3-channel RGB and does not imply that arbitrary RGB imagery is equivalent to the dataset's fixed-stretch spectral composite. The pipeline resizes the 512×512 source chips to 384×384 with bilinear interpolation and normalizes the three stored channels with mean/std 0.5; it does not reorder channels. Small flame signatures may still be lost. No trained model or metric is currently available.

The upstream [label-batch protocol](https://github.com/MohammadrezaNarimaniUCDavis/California_Sentinel2_Active_Fire_Dataset/blob/main/label_batch/README.md) confirms that Label Studio exports must include both brush PNGs and JSON: the JSON identifies reviewed-empty chips and distinguishes active-fire strokes from uncertain ones. The official public repository tree contains the protocol and 233 hand masks, but no completed Label Studio export JSON. Those masks are from the held-out test fires; even recovering their companion JSON would not supply reviewed negatives for the training and validation fires. The California dataset therefore remains unsuitable for this binary classification training run.

## Install

Use Python 3.11 in a virtual environment under `%LOCALAPPDATA%` so the large runtime stays out of the OneDrive project folder; global Python installations are not modified. The supported GPU install below uses an official Windows CUDA wheel. This project was verified with PyTorch 2.12.0 / CUDA 13.0 and the NVIDIA RTX 2050 (4 GB VRAM). NVIDIA's driver must be compatible with the selected PyTorch CUDA wheel.

```powershell
$venv = Join-Path $env:LOCALAPPDATA 'FireWatchAI\venv'
py -3.11 -m venv $venv
& "$venv\Scripts\Activate.ps1"
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install pytest
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

The check must print `True` and `NVIDIA GeForce RTX 2050` before training is enabled. If it does not, the desktop GUI reports the reason and leaves training disabled. There is no silent CPU fallback. Inference can run on CPU if CUDA is unavailable. ImageNet pretrained weights are not downloaded by default.

## Desktop application

Start FireWatch AI locally from the activated environment:

```powershell
python -m firewatch.gui
```

The GUI accepts validated 3-channel RGB JPEG/PNG files, preserves aspect ratio in the preview, and calls the existing FireWatch inference module. It disables detection until a trained and evaluated compatible checkpoint can be loaded. Its dataset and GPU status checks run locally; training runs in a Qt worker thread and can be stopped at a batch boundary. The Python environment, California archive, generated manifests, and default checkpoint are stored under `%LOCALAPPDATA%`, outside Git and the OneDrive project folder.

## Local use

Prepare the approved California archive after extracting it locally. This validates each RGB image/mask pair, checks `fire_px` and `nodata_pct` against actual masks, verifies all fire/count partitions, and skips masks with no valid pixels (and chips above the configurable 50% no-data limit):

```powershell
python -m firewatch.cli prepare-california "$env:LOCALAPPDATA\firewatchAI-data\dataset\California_S2_Active_Fire_Dataset_v1.0"
```

The command writes a weak positive candidate manifest (`image,label,event_id,split,fire_px,nodata_pct,label_source`) and a separate review queue. `ACTIVE_FIRE_EVIDENCE` means the automatic HOTMAP mask has at least one value-1 pixel. Chips with no value-1 pixels are marked `UNCERTAIN_OR_UNSUPPORTED` in the review queue, not used as binary negatives: the masks have no cloud class, and valid background can include cloud-obscured pixels. The adapter therefore reports `training_ready: false`. Mask value 255 is ignored as no-data. The dataset's official fire-held-out splits are preserved. The included 233 analyst brush-mask PNGs are not treated as a complete classification test set because the archive lacks the accompanying JSON that identifies reviewed empty and uncertain chips. Training requires an appropriately reviewed negative class; until then, the generic training pipeline is implemented but no model can honestly be trained from this dataset alone.

Check an image manifest (CSV columns: `image,label,event_id`; optional `split` must be `train`, `validation`, or `test`; image paths may be absolute or relative to the CSV):

```powershell
python -m firewatch.cli inspect path\to\manifest.csv
```

Labels must be `ACTIVE_FIRE_EVIDENCE` or `NO_VISIBLE_FIRE_EVIDENCE`, and each row must name an event. The validator rejects missing files, invalid labels, and byte-identical duplicate files. `UNCERTAIN_OR_UNSUPPORTED` is an inference outcome, not a trainable label in this two-class pipeline. Label provenance must be established by the dataset provider; do not create labels from event association, hotspot proximity, burn scars, or absence of a satellite detection.

Training refuses to start unless both classes appear in all event-disjoint train, validation, and held-out test splits, each row documents a label source, and labels are not the weak HOTMAP mask-presence labels. The splits must use disjoint fire events. If no split column is given, deterministic event-disjoint train/validation/test splits (70/15/15 by event, seed 42 by default) are created. Training explicitly requires `cuda`; CUDA unavailability raises an error without trying CPU. Validation loss selects the checkpoint, and evaluation is recorded on the held-out test partition. Metrics reflect only the provenance and quality of the supplied labels.

```powershell
python -m firewatch.cli train path\to\manifest.csv --epochs 5 --batch-size 2
```

Rows need `label_source` values describing human-reviewed or provider-verified labels. The California weak-positive manifest cannot pass these checks. The local GUI uses batch size 2 by default for the 4 GB GPU; reduce it to 1 if CUDA reports an out-of-memory error.

Evaluate a saved checkpoint on the manifest's deterministic test split:

```powershell
python -m firewatch.cli evaluate path\to\manifest.csv artifacts\firewatch-resnet18.pt
```

Run inference; default abstention threshold is 0.70 and may be set with `--threshold` (valid range 0.5–1.0):

```powershell
python -m firewatch.cli predict path\to\image.png --checkpoint artifacts\firewatch-resnet18.pt --threshold 0.70
```

When weights are absent or input is unsupported, inference returns `UNCERTAIN_OR_UNSUPPORTED`; it never supplies a random or hardcoded class. The score is an uncalibrated softmax score, not a probability of real-world fire. Calibration and a separate evaluation set are needed before interpreting it probabilistically.

Supported file formats are RGB JPEG and PNG. Grayscale, alpha/multichannel PNG, TIFF, corrupted files, images smaller than 32×32, and images larger than 40 million pixels are rejected. No geographic coordinates are inferred. Preprocessing resizes to 384×384 with bilinear interpolation and applies fixed per-channel normalization; multispectral GeoTIFF support is not implemented. Ordinary visible-light RGB input lacks the SWIR/NIR bands present in the training-candidate corpus and is experimental only.

## Tests

```powershell
python -m pytest
```

## Implementation status

Implemented: strict image loading, manifest and label checks, duplicate detection, California mask validation and weak-positive extraction, a separate uncertain review queue for mask-zero chips, preservation/validation of official fire partitions, PyQt6 desktop GUI, asynchronous dataset checks, structured inference with abstention, CUDA status reporting, a CUDA-only sourced-label training guard and progress reporting, checkpoint metadata, held-out evaluation, CLI, and unit tests.

Current model status: no validated trained checkpoint and no measured performance metrics. Training is blocked because the source does not provide reviewed negatives for train and validation. No deployment is configured. The approved dataset archive, extraction, generated manifests, and inspected samples remain in local AppData outside Git.

## Development log

See [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) for phase records, test results, commits, and push status.
