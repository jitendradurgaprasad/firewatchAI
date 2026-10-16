"""Local PyQt6 desktop interface for FireWatch AI."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from time import monotonic
from pathlib import Path
from threading import Event

from PIL import Image
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QDoubleSpinBox, QTextEdit, QVBoxLayout,
    QWidget,
)

from .data import split_by_event, validate_manifest
from .image_io import ImageInputError, load_rgb_image
from .readiness import inspect_california_dataset, inspect_training_manifest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler()])
LOG = logging.getLogger("firewatch.gui")


def default_dataset_path() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    return local / "firewatchAI-data" / "dataset" / "California_S2_Active_Fire_Dataset_v1.0"


def default_checkpoint_path() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    return local / "FireWatchAI" / "artifacts" / "firewatch-resnet18.pt"


def _torch_module():
    import torch
    return torch


def gpu_report() -> dict:
    torch = _torch_module()
    cuda = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if cuda else None
    driver_gpu = "not detected"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                                    check=True, capture_output=True, text=True, timeout=8)
            driver_gpu = result.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            LOG.warning("nvidia-smi query failed: %s", exc)
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda, "device": name,
            "driver_gpu": driver_gpu,
            "status": "CUDA ready" if cuda else "CUDA unavailable to PyTorch"}


class DatasetWorker(QThread):
    completed = pyqtSignal(dict)

    def __init__(self, dataset_root: Path, parent=None):
        super().__init__(parent)
        self.dataset_root = dataset_root

    def run(self):
        try:
            self.completed.emit(inspect_california_dataset(self.dataset_root).to_dict())
        except Exception as exc:
            LOG.exception("Dataset readiness inspection failed")
            self.completed.emit({"available": False, "validation_status": "failed",
                                 "training_ready": False, "location": str(self.dataset_root),
                                 "message": f"Dataset inspection failed: {exc}"})


class PredictionWorker(QThread):
    completed = pyqtSignal(dict)

    def __init__(self, image_path: Path, checkpoint: Path, device: str,
                 threshold: float = 0.7, parent=None):
        super().__init__(parent)
        self.image_path, self.checkpoint, self.device = image_path, checkpoint, device
        self.threshold = threshold

    def run(self):
        try:
            from .inference import predict
            result = predict(self.image_path, self.checkpoint, device=self.device,
                             threshold=self.threshold)
            self.completed.emit(result.to_dict())
        except Exception as exc:
            LOG.exception("Prediction worker failed")
            self.completed.emit({"outcome": "MODEL NOT AVAILABLE", "predicted_class": None,
                                 "confidence_score": None, "message": str(exc),
                                 "limitations": [], "model_id": "firewatch-resnet18",
                                 "model_version": "0.1.0"})


class TrainingWorker(QThread):
    progress = pyqtSignal(dict)
    completed = pyqtSignal(str, dict)
    failed = pyqtSignal(str)
    stopped = pyqtSignal(str)

    def __init__(self, manifest: Path, checkpoint: Path, *, epochs: int,
                 batch_size: int, learning_rate: float, parent=None):
        super().__init__(parent)
        self.manifest, self.checkpoint = manifest, checkpoint
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate = learning_rate
        self.stop_event = Event()

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        try:
            torch = _torch_module()
            from .cli import _evaluate
            from .model import train_model, update_checkpoint_metadata
        except Exception as exc:
            LOG.exception("Could not initialize PyTorch training")
            self.failed.emit(f"Training initialization failed: {exc}")
            return
        try:
            rows = validate_manifest(self.manifest)
            parts = split_by_event(rows)
            batch_size = self.batch_size
            while True:
                try:
                    model = train_model(parts["train"], parts["validation"], self.checkpoint,
                                        epochs=self.epochs, batch_size=batch_size,
                                        learning_rate=self.learning_rate, seed=42, device="cuda",
                                        on_progress=self.progress.emit, stop_event=self.stop_event)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    self.progress.emit({"epoch": 1, "epochs": self.epochs, "batch": 0,
                                        "batches": 1, "device": torch.cuda.get_device_name(0),
                                        "status": "memory_retry", "batch_size": batch_size})
            metrics = _evaluate(model, parts["test"], "cuda")
            update_checkpoint_metadata(self.checkpoint, {"test_metrics": metrics,
                                                         "test_samples": len(parts["test"])})
            metadata = {"checkpoint": str(self.checkpoint), "device": torch.cuda.get_device_name(0),
                        "test_samples": len(parts["test"]), "test_metrics": metrics}
            self.completed.emit("Training and held-out evaluation completed.", metadata)
        except InterruptedError as exc:
            self.stopped.emit(str(exc))
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            LOG.exception("CUDA out of memory during training")
            self.failed.emit("NVIDIA GPU ran out of memory. Training stopped without switching to CPU. "
                             "Retry with a smaller batch size (1 or 2) and close other GPU-heavy apps. "
                             f"Details: {exc}")
        except Exception as exc:
            LOG.exception("Training or evaluation failed")
            self.failed.emit(f"Training/evaluation failed: {exc}")


class GPUWorker(QThread):
    completed = pyqtSignal(dict)

    def run(self):
        try:
            self.completed.emit(gpu_report())
        except Exception as exc:
            LOG.exception("GPU/CUDA status check failed")
            self.completed.emit({"torch": "unavailable", "cuda_runtime": None,
                                 "cuda_available": False, "device": None,
                                 "driver_gpu": "unknown",
                                 "status": f"GPU/CUDA status check failed: {exc}"})


class ModelStatusWorker(QThread):
    completed = pyqtSignal(dict)

    def __init__(self, checkpoint: Path, parent=None):
        super().__init__(parent)
        self.checkpoint = checkpoint

    def run(self):
        try:
            from .model import PREPROCESSING, load_checkpoint
            model, metadata = load_checkpoint(self.checkpoint, device="cpu")
            if metadata.get("preprocessing") != PREPROCESSING:
                raise ValueError("Checkpoint preprocessing metadata is incompatible.")
            if not metadata.get("test_metrics") or int(metadata.get("test_samples", 0)) < 1:
                raise ValueError("Held-out evaluation results are not recorded for this checkpoint.")
            if not str(metadata.get("device", "")).startswith("cuda:0"):
                raise ValueError("Checkpoint metadata does not verify training on the NVIDIA CUDA device.")
            del model
            message = f"Checkpoint verified · version {metadata.get('model_version', 'unknown')} · {self.checkpoint}"
            if metadata.get("device"):
                message += f"\nTraining device: {metadata['device']}"
            message += f"\nHeld-out metrics: {metadata['test_metrics']}"
            self.completed.emit({"valid": True, "message": message})
        except Exception as exc:
            LOG.exception("Checkpoint validation failed")
            self.completed.emit({"valid": False, "message": f"Checkpoint loading failed: {exc}"})


class Card(QFrame):
    def __init__(self, title: str, subtitle: str | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(18, 16, 18, 16)
        self.layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        self.layout.addWidget(heading)
        if subtitle:
            hint = QLabel(subtitle)
            hint.setObjectName("muted")
            hint.setWordWrap(True)
            self.layout.addWidget(hint)


class FireWatchWindow(QMainWindow):
    def __init__(self, dataset_root: str | Path | None = None,
                 checkpoint: str | Path | None = None):
        super().__init__()
        self.dataset_root = Path(dataset_root) if dataset_root else default_dataset_path()
        self.checkpoint = Path(checkpoint) if checkpoint else default_checkpoint_path()
        self.selected_image: Path | None = None
        self._loaded_preview: Image.Image | None = None
        self.dataset_worker = None
        self.prediction_worker = None
        self.training_worker = None
        self.gpu_worker = None
        self.model_status_worker = None
        self.dataset_report: dict = {"available": False, "training_ready": False}
        self.gpu = {"torch": "checking", "cuda_runtime": None, "cuda_available": False,
                    "device": None, "driver_gpu": "checking", "status": "checking"}
        self.setWindowTitle("FireWatch AI — Satellite Image Fire Detection")
        self.resize(1320, 940)
        self._build_ui()
        self._refresh_model_status()
        self._show_gpu_status()
        self._start_gpu_check()
        self._start_dataset_check()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(24, 22, 24, 28)
        root.setSpacing(16)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("FireWatch AI")
        title.setObjectName("appTitle")
        title_box.addWidget(title)
        subtitle = QLabel("Satellite Image Fire Detection  ·  Local engineering prototype")
        subtitle.setObjectName("muted")
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        self.model_badge = QLabel("Checking model…")
        self.model_badge.setObjectName("statusBadge")
        self.model_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.model_badge, 0, Qt.AlignmentFlag.AlignRight)
        root.addLayout(header)

        intro = QLabel("FireWatch classifies image evidence only. It does not infer coordinates or replace operational fire monitoring.")
        intro.setObjectName("notice")
        intro.setWordWrap(True)
        root.addWidget(intro)

        top = QGridLayout()
        top.setSpacing(16)
        image_card = Card("Image inspection", "RGB JPEG/PNG only · file validation is applied before preview or inference")
        self.preview = QLabel("Select an image to preview")
        self.preview.setObjectName("preview")
        self.preview.setMinimumSize(440, 300)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_card.layout.addWidget(self.preview, 1)
        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setObjectName("muted")
        self.image_path_label.setWordWrap(True)
        image_card.layout.addWidget(self.image_path_label)
        image_buttons = QHBoxLayout()
        self.select_image_button = QPushButton("Select Satellite Image")
        self.select_image_button.setObjectName("primaryButton")
        self.select_image_button.clicked.connect(self._select_image)
        self.clear_image_button = QPushButton("Clear image")
        self.clear_image_button.clicked.connect(self._clear_image)
        image_buttons.addWidget(self.select_image_button)
        image_buttons.addWidget(self.clear_image_button)
        image_card.layout.addLayout(image_buttons)
        self.input_status = QLabel("Input validation: waiting for an image")
        self.input_status.setObjectName("muted")
        image_card.layout.addWidget(self.input_status)

        result_card = Card("Detection result", "A valid trained checkpoint is required for a prediction")
        self.result_outcome = QLabel("MODEL NOT AVAILABLE")
        self.result_outcome.setObjectName("resultOutcome")
        self.result_outcome.setWordWrap(True)
        result_card.layout.addWidget(self.result_outcome)
        self.result_detail = QLabel("No validated trained model is currently available.")
        self.result_detail.setWordWrap(True)
        result_card.layout.addWidget(self.result_detail)
        self.confidence = QLabel("Confidence: unavailable")
        self.confidence.setObjectName("muted")
        result_card.layout.addWidget(self.confidence)
        self.result_model_info = QLabel("Model: firewatch-resnet18 · version 0.1.0")
        self.result_model_info.setObjectName("muted")
        result_card.layout.addWidget(self.result_model_info)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Abstention threshold"))
        self.abstention_threshold = QDoubleSpinBox()
        self.abstention_threshold.setRange(0.5, 1.0)
        self.abstention_threshold.setDecimals(2)
        self.abstention_threshold.setSingleStep(0.05)
        self.abstention_threshold.setValue(0.70)
        threshold_row.addWidget(self.abstention_threshold)
        threshold_row.addStretch(1)
        result_card.layout.addLayout(threshold_row)
        self.detect_button = QPushButton("Detect Fire")
        self.detect_button.setObjectName("primaryButton")
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(self._detect)
        result_card.layout.addWidget(self.detect_button)
        self.compatibility_note = QLabel("Dataset imagery is false-colour Sentinel-2: R=B12/SWIR2, G=B11/SWIR1, B=B8A/NIR. Ordinary RGB images do not contain those spectral bands and are experimental inputs only.")
        self.compatibility_note.setObjectName("notice")
        self.compatibility_note.setWordWrap(True)
        result_card.layout.addWidget(self.compatibility_note)
        top.addWidget(image_card, 0, 0)
        top.addWidget(result_card, 0, 1)
        root.addLayout(top)

        status_grid = QGridLayout()
        status_grid.setSpacing(16)
        dataset_card = Card("Dataset readiness")
        self.dataset_status_text = QLabel("Dataset check pending…")
        self.dataset_status_text.setObjectName("statusText")
        self.dataset_status_text.setWordWrap(True)
        dataset_card.layout.addWidget(self.dataset_status_text)
        self.dataset_path_text = QLabel(str(self.dataset_root))
        self.dataset_path_text.setObjectName("muted")
        self.dataset_path_text.setWordWrap(True)
        dataset_card.layout.addWidget(self.dataset_path_text)
        self.refresh_dataset_button = QPushButton("Refresh dataset status")
        self.refresh_dataset_button.clicked.connect(self._start_dataset_check)
        dataset_card.layout.addWidget(self.refresh_dataset_button)
        model_card = Card("Model and GPU")
        self.model_status_text = QLabel()
        self.model_status_text.setObjectName("statusText")
        self.model_status_text.setWordWrap(True)
        model_card.layout.addWidget(self.model_status_text)
        self.gpu_status_text = QLabel()
        self.gpu_status_text.setObjectName("statusText")
        self.gpu_status_text.setWordWrap(True)
        model_card.layout.addWidget(self.gpu_status_text)
        self.gpu_refresh_button = QPushButton("Recheck NVIDIA / CUDA")
        self.gpu_refresh_button.clicked.connect(self._refresh_gpu)
        model_card.layout.addWidget(self.gpu_refresh_button)
        status_grid.addWidget(dataset_card, 0, 0)
        status_grid.addWidget(model_card, 0, 1)
        root.addLayout(status_grid)

        train_card = Card("Training and evaluation", "Training only unlocks for sourced two-class train/validation manifests and a working NVIDIA CUDA device")
        form = QFormLayout()
        manifest_row = QHBoxLayout()
        self.manifest_edit = QLineEdit(str(self.dataset_root / "firewatch_manifest.csv"))
        self.manifest_edit.setPlaceholderText("Path to reviewed-label manifest")
        self.manifest_edit.textChanged.connect(self._check_training_readiness)
        browse_manifest = QPushButton("Choose manifest…")
        browse_manifest.clicked.connect(self._select_manifest)
        manifest_row.addWidget(self.manifest_edit, 1)
        manifest_row.addWidget(browse_manifest)
        form.addRow("Reviewed label manifest", manifest_row)
        options = QHBoxLayout()
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 8)
        self.batch_size.setValue(2)
        self.batch_size.setToolTip("Batch size 2 is conservative for the RTX 2050 4 GB VRAM.")
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(5)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(5)
        self.learning_rate.setRange(0.00001, 0.1)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setValue(0.001)
        self.train_device = QComboBox()
        self.train_device.addItem("NVIDIA RTX 2050 · CUDA", "cuda")
        self.train_device.setEnabled(False)
        options.addWidget(QLabel("Batch")); options.addWidget(self.batch_size)
        options.addWidget(QLabel("Epochs")); options.addWidget(self.epochs)
        options.addWidget(QLabel("Learning rate")); options.addWidget(self.learning_rate)
        options.addWidget(QLabel("Device")); options.addWidget(self.train_device, 1)
        form.addRow("Configuration", options)
        train_card.layout.addLayout(form)
        self.training_ready_text = QLabel("Training blocked until a suitable reviewed manifest is selected.")
        self.training_ready_text.setObjectName("warningText")
        self.training_ready_text.setWordWrap(True)
        train_card.layout.addWidget(self.training_ready_text)
        train_controls = QHBoxLayout()
        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.setObjectName("primaryButton")
        self.start_training_button.setEnabled(False)
        self.start_training_button.clicked.connect(self._start_training)
        self.stop_training_button = QPushButton("Stop Training")
        self.stop_training_button.setEnabled(False)
        self.stop_training_button.clicked.connect(self._stop_training)
        train_controls.addWidget(self.start_training_button)
        train_controls.addWidget(self.stop_training_button)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        train_controls.addWidget(self.training_progress, 1)
        train_card.layout.addLayout(train_controls)
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumHeight(150)
        self.training_log.setPlaceholderText("Training and evaluation messages appear here.")
        train_card.layout.addWidget(self.training_log)
        root.addWidget(train_card)

        footer = QLabel("Local only · No deployment · Softmax scores are uncalibrated and are not probabilities of real-world fire.")
        footer.setObjectName("muted")
        root.addWidget(footer)
        scroll.setWidget(content)
        self.setCentralWidget(scroll)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #101722; color: #e9f0f7; font-family: Segoe UI; font-size: 10pt; }
            QFrame#card { background: #172331; border: 1px solid #2a3b4e; border-radius: 12px; }
            QLabel#appTitle { font-size: 24pt; font-weight: 700; color: #f5f8fc; }
            QLabel#cardTitle { font-size: 13pt; font-weight: 650; color: #f4f7fb; }
            QLabel#muted { color: #9db0c4; }
            QLabel#preview { background: #0c131b; border: 1px dashed #4a6075; border-radius: 8px; color: #8ea2b5; }
            QLabel#statusBadge { background: #382c1d; color: #ffd08a; border: 1px solid #6d532d; padding: 8px 14px; border-radius: 12px; font-weight: 650; }
            QLabel#notice { background: #142d35; color: #b9e7e2; border: 1px solid #2e5c62; padding: 10px; border-radius: 8px; }
            QLabel#resultOutcome { color: #ffd08a; font-size: 17pt; font-weight: 700; padding: 6px 0; }
            QLabel#statusText { color: #d3deea; line-height: 140%; }
            QLabel#warningText { color: #ffd08a; background: #382c1d; padding: 10px; border-radius: 7px; }
            QPushButton { background: #26384a; border: 1px solid #3a5268; border-radius: 7px; padding: 9px 13px; }
            QPushButton:hover:enabled { background: #314a60; }
            QPushButton:disabled { color: #687b8e; background: #1b2632; border-color: #293746; }
            QPushButton#primaryButton { background: #167d77; border: 1px solid #2aa79c; color: white; font-weight: 700; }
            QPushButton#primaryButton:hover:enabled { background: #1a948b; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit { background: #0e1721; border: 1px solid #34495e; border-radius: 6px; padding: 7px; }
            QProgressBar { background: #0d1620; border: 1px solid #34495e; border-radius: 6px; text-align: center; min-height: 22px; }
            QProgressBar::chunk { background: #1b9b8e; border-radius: 5px; }
            QScrollArea { border: 0; }
        """)

    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Satellite Image", "",
                                               "Images (*.jpg *.jpeg *.png)")
        if path:
            self.set_selected_image(path)

    def set_selected_image(self, path: str | Path):
        try:
            loaded = load_rgb_image(path)
        except ImageInputError as exc:
            self.selected_image = None
            self._loaded_preview = None
            self.preview.setText("Image could not be validated")
            self.preview.setPixmap(QPixmap())
            self.image_path_label.setText(str(path))
            self.input_status.setText(f"Input validation: invalid — {exc}")
            self.result_outcome.setText("UNCERTAIN OR UNSUPPORTED IMAGE")
            self.result_detail.setText(str(exc))
            self._update_detection_enabled()
            return False
        self.selected_image = Path(path).resolve()
        self._loaded_preview = loaded.image
        self._loaded_preview.thumbnail((1280, 900), Image.Resampling.LANCZOS)
        self.image_path_label.setText(f"{self.selected_image}  ·  {loaded.width} × {loaded.height} · {loaded.format}")
        self.input_status.setText("Input validation: passed — 3-channel RGB; no geographic coordinates inferred")
        self._render_preview()
        self._update_detection_enabled()
        return True

    def _render_preview(self):
        if self._loaded_preview is None:
            return
        image = self._loaded_preview
        qimage = QImage(image.tobytes("raw", "RGB"), image.width, image.height,
                        image.width * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_preview()

    def _clear_image(self):
        self.selected_image = None
        self._loaded_preview = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Select an image to preview")
        self.image_path_label.setText("No image selected")
        self.input_status.setText("Input validation: waiting for an image")
        self.result_outcome.setText("MODEL NOT AVAILABLE")
        self.result_detail.setText("No validated trained model is currently available.")
        self.confidence.setText("Confidence: unavailable")
        self._update_detection_enabled()

    def _refresh_model_status(self):
        self.model_status_text.setText(f"Model: firewatch-resnet18 · version 0.1.0\n"
                                       f"Architecture: ResNet-18\nCheckpoint: {self.checkpoint}")
        if not self.checkpoint.is_file():
            self.model_valid = False
            self.model_status_text.setText(self.model_status_text.text() +
                "\nTraining: not trained\nNo validated trained model is currently available.")
            self.model_badge.setText("MODEL NOT TRAINED")
            self._update_detection_enabled()
            return
        self.model_valid = False
        self.model_badge.setText("MODEL LOADING")
        self.model_status_text.setText(self.model_status_text.text() + "\nLoading and verifying checkpoint…")
        self.model_status_worker = ModelStatusWorker(self.checkpoint, self)
        self.model_status_worker.completed.connect(self._show_model_status)
        self.model_status_worker.start()

    def _show_model_status(self, result: dict):
        self.model_valid = bool(result.get("valid"))
        self.model_status_text.setText(f"Architecture: ResNet-18 · {result.get('message', 'Model status unavailable.')}")
        training = bool(self.training_worker and self.training_worker.isRunning())
        self.model_badge.setText("TRAINING IN PROGRESS" if training else
                                 ("MODEL READY" if self.model_valid else "ERROR / UNSUPPORTED CHECKPOINT"))
        self._update_detection_enabled()

    def _update_detection_enabled(self):
        self.detect_button.setEnabled(bool(self.selected_image and getattr(self, "model_valid", False)
                                           and not (self.prediction_worker and self.prediction_worker.isRunning())))

    def _detect(self):
        if not self.selected_image:
            self.result_outcome.setText("UNCERTAIN OR UNSUPPORTED IMAGE")
            self.result_detail.setText("Select a valid RGB JPEG or PNG first.")
            return
        if not self.model_valid:
            self.result_outcome.setText("MODEL NOT AVAILABLE")
            self.result_detail.setText("A valid trained checkpoint is required; detection did not run.")
            return
        self.result_outcome.setText("MODEL LOADING / INFERENCE IN PROGRESS")
        self.result_detail.setText("Running the existing FireWatch inference pipeline…")
        device = "cuda" if self.gpu["cuda_available"] else "cpu"
        self.prediction_worker = PredictionWorker(self.selected_image, self.checkpoint, device,
                                                  self.abstention_threshold.value(), self)
        self.prediction_worker.completed.connect(self._show_prediction)
        self.prediction_worker.start()
        self._update_detection_enabled()

    def _show_prediction(self, prediction: dict):
        outcome = prediction.get("outcome", "UNCERTAIN OR UNSUPPORTED IMAGE")
        self.result_outcome.setText(outcome)
        self.result_detail.setText(prediction.get("message", "No prediction details returned."))
        score = prediction.get("confidence_score")
        score_kind = prediction.get("score_kind", "unavailable")
        self.result_model_info.setText(f"Model: {prediction.get('model_id', 'unknown')} · version {prediction.get('model_version', 'unknown')}")
        self.confidence.setText(f"Confidence: {score:.4f} ({score_kind}; not a calibrated probability)"
                                if score is not None else "Confidence: unavailable")
        limitations = prediction.get("limitations", [])
        if limitations:
            self.result_detail.setText(self.result_detail.text() + "\n\nLimitations: " + " ".join(limitations))
        self.input_status.setText(f"Input validation: {prediction.get('input_validation', 'unknown')} · inference device: {prediction.get('inference_device', 'unknown')}")
        self._update_detection_enabled()

    def _start_dataset_check(self):
        self.refresh_dataset_button.setEnabled(False)
        self.dataset_status_text.setText("Validating image/mask pairs and reading weak-label counts…")
        self.dataset_worker = DatasetWorker(self.dataset_root, self)
        self.dataset_worker.completed.connect(self._show_dataset_status)
        self.dataset_worker.start()

    def _show_dataset_status(self, report: dict):
        self.dataset_report = report
        if report.get("validation_status") == "passed":
            lines = [f"Dataset validation: PASSED · {report['validated_pairs']:,} image/mask pairs",
                     f"Weak positive mask candidates: {report['weak_positive_candidates']:,}",
                     f"Reviewed negative examples: {report['reviewed_negative_examples']:,}",
                     f"Uncertain/no-positive mask chips: {report['uncertain_or_excluded']:,}",
                     f"Excluded for no-data: {report['skipped_nodata']:,}",
                     f"Label Studio JSON available locally: {'yes' if report['json_export_available'] else 'no'}",
                     "Training readiness: BLOCKED — no verified negatives in train/validation.",
                     report["message"]]
        else:
            lines = [f"Dataset status: {report.get('validation_status', 'unavailable')}", report.get("message", "Dataset unavailable.")]
        self.dataset_status_text.setText("\n".join(lines))
        self.refresh_dataset_button.setEnabled(True)
        self._check_training_readiness()

    def _show_gpu_status(self):
        status = self.gpu
        if status.get("status") == "checking":
            text = "Checking NVIDIA GPU and PyTorch CUDA in a background worker…"
        elif status.get("status", "").startswith("GPU/CUDA status check failed"):
            text = status["status"]
        elif status["cuda_available"]:
            text = (f"NVIDIA GPU: {status['device']}\nSelected training device: CUDA\n"
                    f"PyTorch: {status['torch']} · CUDA runtime {status['cuda_runtime']}\nCUDA verification: available")
        else:
            text = (f"NVIDIA hardware: {status['driver_gpu']}\nSelected training device: unavailable\n"
                    f"PyTorch: {status['torch']} · bundled CUDA: {status['cuda_runtime']}\n"
                    "CUDA verification: unavailable; training stays disabled and will not fall back to CPU.")
        self.gpu_status_text.setText(text)
        self.train_device.setEnabled(status["cuda_available"])
        self._check_training_readiness()

    def _refresh_gpu(self):
        self._start_gpu_check()

    def _start_gpu_check(self):
        self.gpu_refresh_button.setEnabled(False)
        self.gpu_status_text.setText("Checking NVIDIA GPU and PyTorch CUDA in a background worker…")
        self.gpu_worker = GPUWorker(self)
        self.gpu_worker.completed.connect(self._gpu_status_ready)
        self.gpu_worker.start()

    def _gpu_status_ready(self, report: dict):
        self.gpu = report
        self._show_gpu_status()
        self.gpu_refresh_button.setEnabled(True)

    def _select_manifest(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose reviewed label manifest", "",
                                               "CSV manifests (*.csv)")
        if path:
            self.manifest_edit.setText(path)

    def _check_training_readiness(self):
        manifest_path = Path(self.manifest_edit.text()).expanduser()
        california_manifest = self.dataset_root.resolve() / "firewatch_manifest.csv"
        if manifest_path.resolve() == california_manifest:
            self.training_ready_text.setText("California source manifest contains weak positives only; mask-zero chips are uncertain, not negatives.")
            ready = False
        elif manifest_path.is_file():
            ready, message, counts = inspect_training_manifest(manifest_path)
            self.training_ready_text.setText(message + (f" · split samples: {counts}" if counts else ""))
        else:
            ready = False
            self.training_ready_text.setText("Training blocked until a suitable reviewed manifest is selected.")
        if not self.gpu.get("cuda_available"):
            ready = False
            self.training_ready_text.setText(self.training_ready_text.text() + " NVIDIA CUDA is unavailable.")
        busy = bool(self.training_worker and self.training_worker.isRunning())
        self.start_training_button.setEnabled(bool(ready and not busy))
        self.stop_training_button.setEnabled(busy)

    def _start_training(self):
        ready, message, _ = inspect_training_manifest(self.manifest_edit.text())
        if not ready or not self.gpu.get("cuda_available"):
            self.training_ready_text.setText(message + " Training remains blocked.")
            return
        manifest = Path(self.manifest_edit.text()).expanduser().resolve()
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.training_worker = TrainingWorker(manifest, self.checkpoint,
            epochs=self.epochs.value(), batch_size=self.batch_size.value(),
            learning_rate=self.learning_rate.value(), parent=self)
        self.training_worker.progress.connect(self._training_progress)
        self.training_worker.completed.connect(self._training_completed)
        self.training_worker.failed.connect(self._training_failed)
        self.training_worker.stopped.connect(self._training_stopped)
        self.training_worker.finished.connect(self._training_thread_finished)
        self.training_log.append(f"Starting ResNet-18 training on {self.gpu['device']} with batch size {self.batch_size.value()}.")
        self.training_progress.setValue(0)
        self.training_worker.start()
        self._check_training_readiness()

    def _stop_training(self):
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.request_stop()
            self.stop_training_button.setEnabled(False)
            self.training_log.append("Stop requested; finishing the current batch first.")

    def _training_progress(self, report: dict):
        if report.get("status") == "memory_retry":
            self.batch_size.setValue(report["batch_size"])
            self.training_log.append(f"CUDA out of memory; clearing cache and retrying with batch size {report['batch_size']}.")
            self.training_ready_text.setText(f"GPU memory recovery: retrying with batch size {report['batch_size']} on {report['device']}.")
            return
        epochs, epoch = report["epochs"], report["epoch"]
        batches, batch = max(1, report["batches"]), report["batch"]
        pct = int(100 * ((epoch - 1) + batch / batches) / epochs)
        self.training_progress.setValue(min(99, pct))
        detail = f"Epoch {epoch}/{epochs} · batch {batch}/{batches} · CUDA device: {report['device']}"
        if "validation_loss" in report:
            detail += f" · validation loss: {report['validation_loss']:.5f}"
        self.training_ready_text.setText(detail)

    def _training_completed(self, message: str, report: dict):
        self.training_progress.setValue(100)
        self.training_log.append(message + "\n" + str(report))
        self.dataset_status_text.setText(self.dataset_status_text.text() + "\nTraining complete; held-out test evaluation recorded.")
        self._refresh_model_status()
        self._check_training_readiness()

    def _training_failed(self, message: str):
        self.training_log.append(message)
        self.training_ready_text.setText(message)
        self._check_training_readiness()

    def _training_stopped(self, message: str):
        self.training_log.append(message)
        self.training_ready_text.setText("Training stopped at a batch boundary. Partial best checkpoint, if any, must still pass model validation.")
        self._refresh_model_status()
        self._check_training_readiness()

    def _training_thread_finished(self):
        self._refresh_model_status()
        self._check_training_readiness()

    def closeEvent(self, event):
        workers = [worker for worker in (self.dataset_worker, self.prediction_worker, self.training_worker,
                                         self.gpu_worker, self.model_status_worker)
                   if worker is not None and worker.isRunning()]
        if self.training_worker in workers:
            self.training_worker.request_stop()
        still_running = []
        deadline = monotonic() + 5.0
        for worker in workers:
            remaining_ms = max(0, int((deadline - monotonic()) * 1000))
            if not worker.wait(remaining_ms):
                still_running.append(worker)
        if still_running:
            self.training_log.append("Close delayed while a background operation finishes safely; close again in a moment.")
            event.ignore()
            return
        event.accept()


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    app.setApplicationName("FireWatch AI")
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#101722"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e9f0f7"))
    app.setPalette(palette)
    window = FireWatchWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
