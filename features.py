"""ESC-50 feature extraction matching the retained CNN's input pipeline."""

import os

import librosa
import numpy as np

from data_loader import load_metadata


SAMPLE_RATE = 22050
N_MFCC = 128


def extract_mfcc(file_path, n_mfcc=N_MFCC):
    """Return one 128-value mean-MFCC vector, as used to train AudioCNN."""
    waveform, sample_rate = librosa.load(
        file_path, sr=SAMPLE_RATE, mono=True
    )
    if waveform.size == 0:
        raise ValueError(f"Audio file contains no samples: {file_path}")
    coefficients = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=n_mfcc
    )
    return np.mean(coefficients, axis=1).astype(np.float32)


def build_feature_dataset(
    output_x="features_X.npy", output_y="features_y.npy"
):
    """Build aligned feature/target arrays from the ESC-50 metadata CSV."""
    metadata = load_metadata()
    features, targets = [], []
    for row_number, row in metadata.iterrows():
        if not os.path.isfile(row["file_path"]):
            raise FileNotFoundError(
                f"ESC-50 audio file not found: {row['file_path']}. "
                "Download and extract the dataset before feature extraction."
            )
        features.append(extract_mfcc(row["file_path"]))
        targets.append(int(row["target"]))
        if (row_number + 1) % 200 == 0:
            print(f"Processed {row_number + 1} files")
    feature_array = np.asarray(features, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.int64)
    np.save(output_x, feature_array)
    np.save(output_y, target_array)
    print("Saved:", feature_array.shape, target_array.shape)


if __name__ == "__main__":
    build_feature_dataset()
