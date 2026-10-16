# FireWatch AI — satellite image fire evidence

**Current phase:** satellite image fire detection (local research prototype). **Deployment is not permitted for this phase.**

FireWatch AI accepts a JPEG or PNG and is designed to return one of:

- `FIRE EVIDENCE DETECTED`
- `NO VISIBLE FIRE EVIDENCE DETECTED`
- `UNCERTAIN_OR_UNSUPPORTED`

This phase is image classification only. It does not locate or segment fire, match satellite hotspots, fuse data sources, or support emergency response. It cannot establish that every real-world fire will be detected.

## Dataset investigation and attribution

### Candidate review (2026-10-16)

The current classifier contract is a manifest with `image,label,event_id`, optional `split` (`train`, `validation`, `test`), and `label_source`; images must decode as 3-channel RGB JPEG/PNG (the pipeline resizes to 384×384). Labels are `ACTIVE_FIRE_EVIDENCE` and `NO_VISIBLE_FIRE_EVIDENCE`. Source/event-disjoint partitions and documented label provenance are required. Existing preparation command: `python -m firewatch.cli prepare-california <dataset-root>`; generic manifest inspection: `python -m firewatch.cli inspect <manifest.csv>`. There is no adapter for arbitrary multispectral satellite products.

The candidates below are different sensing targets and are not interchangeable:

| Candidate | Modality, documented labels and scale | Licence / evaluation / main limitation | Assessment |
|---|---|---|---|
| [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) | Ground-level, aerial and surveillance RGB; 21,527 images: 1,164 fire only, 5,867 smoke only, 4,658 fire+smoke, 9,838 neither. YOLO object boxes. | Repository says CC0 for the collection but also says it does not own copyright in the underlying images and disclaims clearing third-party rights. The published random image split has documented near-duplicate/source-video leakage; no source-grouped independent test is supplied. Download size is not published in the official repository. | Useful only as a ground-RGB pipeline candidate after rights and source grouping are resolved; not evidence for satellite performance. Not downloaded. |
| [Sen2Fire](https://zenodo.org/records/10881058) | Sentinel-2 + Sentinel-5P multispectral patches, 13 bands, 2,466 patches; labels are fire/non-fire derived from MOD14A1 V6.1. Split by four study areas: 1,458/504/504 train/validation/test. | 6.3 GB. Zenodo does not expose a usable licence declaration on the record checked, so reuse rights could not be verified. MODIS active-fire labels are thermal-anomaly detections, not human-confirmed visible flame labels. Multi-band `.npz` is incompatible with the RGB loader. | Satellite-relevant but cannot be adopted pending licence verification and a compatible spectral/label definition. Not downloaded. |
| [OHID-FF](https://doi.org/10.6084/m9.figshare.28218485.v6) | Zhuhai-1 hyperspectral satellite scenes (32 bands, 10 m); 1,197 PNG chips with YOLO-format boxes and a binary filename suffix. Article reports 647 fire / 550 background chips. | Figshare declares CC BY 4.0; archive is 2,093,569,123 bytes. Published classification split is random across tiles (597/600), not source-scene-disjoint. The released class name and count conflict with the article (details below). | Best satellite-domain candidate to investigate further, but failed label verification and is not training-ready. |
| [California Sentinel-2 Active-Fire Segmentation](https://doi.org/10.5281/zenodo.22713948) | Satellite false-colour SWIR/NIR (B12/B11/B8A), 2,148 512×512 chips across 25 fires; automatic modified-HOTMAP mask positives; incident-held-out 18/3/4 split. | CC BY 4.0; 937,057,654 bytes. 841 weak-positive candidate chips and 1,307 mask-zero uncertain chips; mask zero is not a reviewed negative. | Closest modality/label target, but lacks usable reviewed train/validation negatives; training remains blocked. |

OHID-FF was downloaded from its official Figshare file URL to `%LOCALAPPDATA%\FireWatchAI-data\datasets\OHID-FF.zip`, outside Git. The downloaded 2,093,569,123-byte archive matched the Figshare MD5 `8f70070718e4d0ad7c18517bd6058770`; ZIP CRC validation passed and the archive contained no unsafe paths. Extraction is under `%LOCALAPPDATA%\FireWatchAI-data\datasets\OHID-FF\`. The extracted `YOLODataset/images` and `labels` folders contain 1,197 one-to-one PNG/TXT pairs; all PNGs decoded successfully as RGB, with dimensions between 448 and 512 pixels per side, and no exact duplicate image hashes were found. Filename suffixes yield 652 positive and 545 negative chips, not the article's 647/550. More seriously, `YOLODataset/classes.txt` says `smoke`, while the article describes the sole annotation class as forest-fire points. Since positive label files use class id `0`, the released labels do not resolve this semantic disagreement. The official notebook creates stratified tile-level splits; filenames identify source scenes, so scene grouping is possible, but the article's random 597/600 evaluation split is not independent by source scene. No manifest was generated and no training guard was changed.

**Recommendation:** do not select OHID-FF as a training dataset until the authors clarify the class-name/count inconsistency and a source-scene split is rebuilt and audited. It is satellite imagery and can support satellite-domain development only after those issues are resolved; its PNGs are RGB renderings of hyperspectral data, not ordinary visible-light RGB and not compatible evidence that the existing three-channel ResNet learns the full 32-band signal. For A, it can exercise RGB file loading only, not fire-label learning until clarified. For B, a future model would need the 32-band inputs or a clearly specified, fixed spectral-to-three-channel rendering and independent source-scene evaluation. Accordingly, **no new dataset is currently training-ready**. No download was made for D-Fire or Sen2Fire because of the rights caveat and unresolved licence, respectively.

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
