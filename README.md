# Audio-event analysis and augmentation for VR music production

A Streamlit learning and creative-audio platform for environmental sound
classification, signal visualization, audio enhancement, and multi-track
composition. It retains the original ESC-50/PyTorch classifier and can
optionally use Google's YAMNet model as a separate 521-class backend.

## Community-service objective and SDG alignment

The project is intended to help students and community groups explore
environmental recordings and learn how audio signals and machine-learning
classifiers work. It supports:

- **SDG 4 - Quality education:** interactive audio and model-output
  visualizations for practical learning.
- **SDG 9 - Industry, innovation and infrastructure:** a modular example of
  audio preprocessing, model inference, and creative production tools.
- **SDG 11 - Sustainable cities and communities:** activities for listening to
  and discussing everyday community soundscapes.
- **SDG 15 - Life on land:** educational listening around birds and animals.

These are educational aims, not claims of measured conservation, urban-noise,
health, or learning outcomes.

## Features

- WAV and MP3 upload; OGG and FLAC are also accepted when decodable by the
  installed libsndfile.
- Browser microphone recording through Streamlit's audio input.
- Mono conversion, finite-sample checks, resampling, and overlapping
  five-second model segments.
- Selectable local ESC-50 CNN and optional YAMNet backends.
- Top-five class scores, segment-level candidates, and low-confidence warnings.
- Actual waveform, frequency spectrum, and mel-spectrogram visualizations.
- Session-only analysis history and environmental category summaries.
- Non-destructive gain, echo, reverb, pitch, speed, and filter controls.
- Independent track levels, audio mix preview, and WAV export.
- Dark Streamlit theme and separate home, detection, monitoring, enhancement,
  VR-studio, and SDG sections.

## Technology

- Python 3.11
- Streamlit, PyTorch, librosa, NumPy, SoundFile, SciPy, Pandas, and Plotly
- Optional TensorFlow 2.x and TensorFlow Hub for YAMNet

## Architecture

```text
app.py                    Streamlit UI and session-only history
audio_processing.py       Decode/resample, segmentation, effects, and mixdown
audio_classification.py   ESC-50 CNN and optional YAMNet inference
model.py                  Original compact 50-class PyTorch architecture
model_50class.pth         Preserved original checkpoint (unchanged)
esc50_labels.json         Verified output-index to ESC-50 label mapping
data_loader.py            ESC-50 metadata, dataset, and training split
features.py               Feature extraction matching the CNN's 128-MFCC input
train.py                  Original training loop; writes a new checkpoint name
predict_one.py             Single-file command-line prediction
realtime_mic.py            Continuous microphone command-line prediction
tests/                    Focused audio and inference unit tests
```

### Classification behavior

The preserved checkpoint accepts 128 mean MFCC features calculated from mono
22,050 Hz audio. In the initial implementation, training and inference both
used `n_mfcc=128`, and the sorted ESC-50 `target` index maps correctly to the
checkpoint's 50 output neurons. The identified weaknesses were loss of event
timing from mean pooling, training on only 2,000 ESC-50 clips without
augmentation, and interpreting an uncalibrated softmax maximum as a
high-confidence fact. The training audio directory is not included in this
workspace, so retraining and a measured accuracy claim are not possible.

The updated app segments recordings into overlapping five-second windows,
scores each window, and averages their scores. A close top-two result or a
leading score below the user-selected warning threshold is explicitly marked
uncertain. Scores are still model estimates, not calibrated probabilities.
For short clips, zero-padding supplies the model's expected input duration and
the app warns that results may be unreliable.

The YAMNet option is independent of the retained CNN. It loads TensorFlow
Hub's YAMNet model and obtains labels from that model's own class-map artifact;
the app validates that the map has 521 contiguous output indices. It resamples
mono audio to 16 kHz, segments longer recordings, and aggregates frame scores.
YAMNet is optional and is not included in the default requirements because its
TensorFlow runtime is substantially larger and platform support varies.
TensorFlow Hub needs internet access on first use to retrieve the model.

### Dataset information

The local model is associated with the ESC-50 environmental sound dataset:
2,000 labeled five-second recordings across 50 classes. This workspace has the
ESC-50 metadata CSV but does not contain `data/ESC50/audio/`; therefore it can
load the existing checkpoint and mapping but cannot retrain or run a dataset
evaluation. The app uses `esc50_labels.json` if the metadata CSV is unavailable.

## Installation

From the project directory on Windows:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents environment activation, run the interpreter directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

To try the optional YAMNet backend, install its dependencies separately:

```powershell
pip install -r requirements-yamnet.txt
```

If TensorFlow has no compatible wheel for your Python/Windows combination, use
a TensorFlow-supported Python environment (for example, Linux or WSL) rather
than replacing the default local CNN installation.

## Run the application

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`. Microphone recording requires browser permission.
WAV/MP3 decoding depends on the libsndfile capabilities of the installed
SoundFile package; unsupported or corrupt files produce a readable error.

## Tests

With the project's installed environment:

```powershell
python -m unittest discover -s tests -v
python -m py_compile app.py audio_processing.py audio_classification.py model.py data_loader.py features.py train.py
```

Tests cover decoding/resampling, malformed input, segmentation, non-destructive
effects, WAV export, label-index compatibility, aggregation, and CNN inference.
The optional YAMNet model download and live browser microphone cannot be tested
without TensorFlow Hub and browser/device interaction.

## Model limitations

- ESC-50 has limited examples per class and does not cover every sound or
  recording condition.
- The compact CNN mean-pools MFCCs and has no time-aware architecture or
  probability calibration. Segment averaging retains more context but does not
  retrain or otherwise improve the CNN's learned accuracy.
- A top-ranked label is not a confirmed event; similar classes, background
  noise, microphone characteristics, and unseen sources can change results.
- YAMNet's broad AudioSet categories are not precise wildlife species labels.
- History/statistics are held only in Streamlit session state and are not
  persisted across sessions.
- The enhancement controls are lightweight effects, not a mastering chain.
- The VR studio exports a standard mixed WAV. It does not create spatial audio,
  Unity assets, or Unreal Engine projects yet.

## Dataset training

Training requires the full ESC-50 dataset with the audio files under
`data/ESC50/audio/` and metadata at `data/ESC50/meta/esc50.csv`. The existing
training architecture is preserved. Running `train.py` writes
`model_50class_retrained.pth` so it cannot overwrite the original
`model_50class.pth`. Compare and explicitly validate any retrained checkpoint
before configuring the app to use it.

## File change summary

Implementation changes in this update:

- Modified: `app.py`, `.streamlit/config.toml`, `requirements.txt`, `train.py`,
  `data_loader.py`, `features.py`, `predict_one.py`, `realtime_mic.py`, and
  `README.md`.
- Added: `audio_processing.py`, `audio_classification.py`,
  `esc50_labels.json`, `requirements-yamnet.txt`,
  `tests/test_audio_processing.py`, and `tests/test_audio_classification.py`.
- Preserved: `model.py` and the original `model_50class.pth` checkpoint.

The original project snapshot, including its model, is in the
`original_backup_20261122_195508/` directory.
