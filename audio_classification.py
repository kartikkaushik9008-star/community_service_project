"""Optional YAMNet and retained ESC-50 CNN inference backends."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch

from audio_processing import AudioProcessingError, segment_audio
from model import AudioCNN


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "model_50class.pth"
METADATA_PATH = ROOT / "data" / "ESC50" / "meta" / "esc50.csv"
LABELS_PATH = ROOT / "esc50_labels.json"
CNN_SAMPLE_RATE = 22050
YAMNET_SAMPLE_RATE = 16000
N_MFCC = 128


class ClassifierUnavailableError(RuntimeError):
    """Raised when an optional or local classification backend is unavailable."""


def load_esc50_labels(
    metadata_path: Path = METADATA_PATH, labels_path: Path = LABELS_PATH
) -> dict[int, str]:
    """Load the checkpoint's ESC-50 output-index labels and validate their order."""
    if metadata_path.is_file():
        metadata = pd.read_csv(metadata_path)
        required = {"target", "category"}
        if not required.issubset(metadata.columns):
            raise ClassifierUnavailableError(
                "ESC-50 metadata must contain target and category columns."
            )
        labels = metadata.groupby("target")["category"].first().sort_index().to_dict()
        labels = {int(index): str(label) for index, label in labels.items()}
    elif labels_path.is_file():
        with labels_path.open("r", encoding="utf-8") as file:
            labels = {int(index): str(label) for index, label in json.load(file).items()}
    else:
        raise ClassifierUnavailableError(
            "ESC-50 label metadata is missing. Restore data/ESC50/meta/esc50.csv "
            "or esc50_labels.json."
        )

    if len(labels) != 50 or sorted(labels) != list(range(50)):
        raise ClassifierUnavailableError(
            "ESC-50 labels do not match the checkpoint's 50 output indices."
        )
    return labels


def load_esc50_model(
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> tuple[AudioCNN, dict[int, str]]:
    """Load the original 50-class checkpoint without modifying it."""
    if not checkpoint_path.is_file():
        raise ClassifierUnavailableError(
            f"Local ESC-50 checkpoint not found: {checkpoint_path.name}"
        )
    labels = load_esc50_labels()
    model = AudioCNN(num_classes=50)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ClassifierUnavailableError(
            f"Could not load {checkpoint_path.name}: {exc}"
        ) from exc
    model.eval()
    return model, labels


def _top_predictions(
    scores: np.ndarray, labels: dict[int, str], top_k: int = 5
) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size != len(labels):
        raise ClassifierUnavailableError(
            "Classifier outputs do not match the supplied class-label mapping."
        )
    indices = np.argsort(values)[::-1][:top_k]
    return [
        {"label": labels[int(index)], "confidence": float(values[index])}
        for index in indices
    ]


def _build_result(
    per_segment_scores: list[np.ndarray],
    segment_starts: list[float],
    labels: dict[int, str],
    threshold: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not per_segment_scores:
        raise ClassifierUnavailableError(
            "No segment contained enough signal to classify."
        )
    combined = np.mean(np.stack(per_segment_scores), axis=0)
    top = _top_predictions(combined, labels)
    margin = top[0]["confidence"] - top[1]["confidence"] if len(top) > 1 else 1.0
    timeline = []
    for start, scores in zip(segment_starts, per_segment_scores):
        segment_top = _top_predictions(scores, labels, top_k=1)[0]
        timeline.append(
            {
                "start_seconds": start,
                "label": segment_top["label"],
                "confidence": segment_top["confidence"],
            }
        )
    return {
        "predictions": top,
        "segments": timeline,
        "low_confidence": (
            top[0]["confidence"] < threshold or margin < 0.05
        ),
        "analysis_seconds": elapsed_seconds,
    }


def predict_esc50(
    samples: np.ndarray,
    sample_rate: int,
    model: AudioCNN,
    labels: dict[int, str],
    confidence_threshold: float = 0.30,
) -> dict[str, Any]:
    """Classify 5-second overlapping segments with the original ESC-50 CNN."""
    if sample_rate <= 0:
        raise AudioProcessingError("The audio sample rate must be positive.")
    if sample_rate != CNN_SAMPLE_RATE:
        samples = librosa.resample(
            np.asarray(samples, dtype=np.float32),
            orig_sr=sample_rate,
            target_sr=CNN_SAMPLE_RATE,
        )
        sample_rate = CNN_SAMPLE_RATE

    started = time.perf_counter()
    segments = segment_audio(samples, sample_rate)
    scores = []
    starts = []
    with torch.inference_mode():
        for start, chunk in segments:
            if float(np.sqrt(np.mean(chunk**2))) < 0.001:
                continue
            mfcc = librosa.feature.mfcc(
                y=chunk, sr=sample_rate, n_mfcc=N_MFCC
            )
            features = np.mean(mfcc, axis=1).astype(np.float32)
            logits = model(torch.from_numpy(features).view(1, 1, N_MFCC))
            scores.append(torch.softmax(logits, dim=1).squeeze(0).cpu().numpy())
            starts.append(start)
    return _build_result(
        scores,
        starts,
        labels,
        confidence_threshold,
        time.perf_counter() - started,
    )


def load_yamnet() -> tuple[Any, dict[int, str]]:
    """Load YAMNet from TensorFlow Hub and use its own class-map artifact."""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as exc:
        raise ClassifierUnavailableError(
            "YAMNet is optional and is not installed. Install compatible "
            "TensorFlow 2.x and tensorflow-hub packages, or select the local "
            "ESC-50 CNN. TensorFlow Hub also needs internet access on first use."
        ) from exc

    try:
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = model.class_map_path()
        if hasattr(class_map_path, "numpy"):
            class_map_path = class_map_path.numpy()
        if isinstance(class_map_path, bytes):
            class_map_path = class_map_path.decode("utf-8")
        class_map = pd.read_csv(tf.io.gfile.GFile(class_map_path))
        name_column = "display_name" if "display_name" in class_map else "name"
        labels = {
            int(row["index"]): str(row[name_column])
            for _, row in class_map.iterrows()
        }
        if len(labels) != 521 or sorted(labels) != list(range(521)):
            raise ClassifierUnavailableError(
                "YAMNet class map does not match its 521 output categories."
            )
        return model, labels
    except ClassifierUnavailableError:
        raise
    except Exception as exc:
        raise ClassifierUnavailableError(
            f"Could not load YAMNet or its class map: {exc}"
        ) from exc


def predict_yamnet(
    samples: np.ndarray,
    sample_rate: int,
    model: Any,
    labels: dict[int, str],
    confidence_threshold: float = 0.15,
) -> dict[str, Any]:
    """Classify overlapping segments at YAMNet's required 16 kHz sample rate."""
    if sample_rate <= 0:
        raise AudioProcessingError("The audio sample rate must be positive.")
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate != YAMNET_SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=YAMNET_SAMPLE_RATE
        ).astype(np.float32)
    started = time.perf_counter()
    segments = segment_audio(audio, YAMNET_SAMPLE_RATE)
    scores = []
    starts = []
    for start, chunk in segments:
        if float(np.sqrt(np.mean(chunk**2))) < 0.001:
            continue
        output = model(chunk)
        frame_scores = output[0].numpy() if hasattr(output[0], "numpy") else output[0]
        scores.append(np.mean(np.asarray(frame_scores), axis=0))
        starts.append(start)
    return _build_result(
        scores,
        starts,
        labels,
        confidence_threshold,
        time.perf_counter() - started,
    )
