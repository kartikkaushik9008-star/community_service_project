import os
import pandas as pd
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "ESC50", "meta", "esc50.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "ESC50", "audio")

class ESC50Dataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, audio_dir=AUDIO_DIR, sr=22050, n_mfcc=128):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mfcc = n_mfcc

        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(
                f"ESC-50 metadata not found at {self.csv_path}. "
                "Restore data/ESC50/meta/esc50.csv before training."
            )
        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(
                f"ESC-50 audio directory not found at {self.audio_dir}. "
                "Download and extract the ESC-50 audio files before training."
            )
        self.data = load_metadata(self.csv_path, self.audio_dir)
        self.targets = self.data["target"].values

        self.idx_to_class = (
            self.data.groupby("target")["category"]
            .first()
            .sort_index()
            .to_dict()
        )

        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row["filename"]
        label = int(row["target"])

        path = os.path.join(self.audio_dir, filename)

        waveform, sr = librosa.load(path, sr=self.sr, mono=True)

        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=self.n_mfcc)
        feat = np.mean(mfcc, axis=1).astype(np.float32)

        feat_tensor = torch.from_numpy(feat).unsqueeze(0)

        return feat_tensor, torch.tensor(label, dtype=torch.long)


def load_metadata(csv_path=CSV_PATH, audio_dir=AUDIO_DIR):
    """Load ESC-50 metadata and expose paths/labels shared by project tools."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"ESC-50 metadata not found at {csv_path}. "
            "Restore data/ESC50/meta/esc50.csv before processing the dataset."
        )
    metadata = pd.read_csv(csv_path)
    required = {"filename", "target", "category"}
    if not required.issubset(metadata.columns):
        missing = ", ".join(sorted(required.difference(metadata.columns)))
        raise ValueError(f"ESC-50 metadata is missing required columns: {missing}")
    metadata = metadata.copy()
    metadata["file_path"] = metadata["filename"].map(
        lambda filename: os.path.join(audio_dir, filename)
    )
    metadata["classID"] = metadata["target"].astype(int)
    return metadata


def load_train_val_indices(csv_path=CSV_PATH):
    """Create the deterministic stratified split used by the training script."""
    metadata = load_metadata(csv_path)
    return train_test_split(
        range(len(metadata)),
        test_size=0.2,
        stratify=metadata["target"],
        random_state=42,
    )
