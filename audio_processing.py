"""Audio decoding, visualization inputs, effects, and mixdown helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".ogg", ".flac"}


class AudioProcessingError(ValueError):
    """Raised when an audio file cannot be decoded or processed."""


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray
    sample_rate: int
    source_sample_rate: int


def decode_audio(
    audio_bytes: bytes, filename: str, target_sample_rate: int = 22050
) -> DecodedAudio:
    """Decode supported audio bytes to finite, mono float32 samples."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_FORMATS:
        raise AudioProcessingError(
            f"Unsupported audio format '{suffix or 'unknown'}'. "
            "Upload WAV, MP3, OGG, or FLAC audio."
        )
    if not audio_bytes:
        raise AudioProcessingError("The uploaded audio file is empty.")
    if target_sample_rate <= 0:
        raise AudioProcessingError("The target sample rate must be positive.")

    try:
        with sf.SoundFile(BytesIO(audio_bytes)) as audio_file:
            source_sample_rate = int(audio_file.samplerate)
            samples = audio_file.read(dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioProcessingError(
            "Could not decode this audio file. Check that it is a valid WAV, "
            "MP3, OGG, or FLAC file; MP3 decoding may require a compatible "
            "libsndfile build."
        ) from exc

    if samples.size == 0 or samples.shape[0] == 0:
        raise AudioProcessingError("The uploaded audio file contains no samples.")
    if source_sample_rate <= 0:
        raise AudioProcessingError("The audio file has an invalid sample rate.")

    mono = np.mean(samples, axis=1, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(mono)):
        raise AudioProcessingError("The audio contains invalid numeric samples.")
    mono = np.clip(mono, -1.0, 1.0)
    if source_sample_rate != target_sample_rate:
        try:
            mono = librosa.resample(
                mono, orig_sr=source_sample_rate, target_sr=target_sample_rate
            ).astype(np.float32)
        except Exception as exc:
            raise AudioProcessingError(
                f"Could not resample audio from {source_sample_rate} Hz to "
                f"{target_sample_rate} Hz: {exc}"
            ) from exc
    if mono.size == 0:
        raise AudioProcessingError("The audio is too short to process.")

    return DecodedAudio(mono, target_sample_rate, source_sample_rate)


def segment_audio(
    samples: np.ndarray,
    sample_rate: int,
    window_seconds: float = 5.0,
    hop_seconds: float = 2.5,
) -> list[tuple[float, np.ndarray]]:
    """Return fixed-size overlapping model segments and their start times."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or window_seconds <= 0 or hop_seconds <= 0:
        raise AudioProcessingError("Invalid sample rate or segment duration.")
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise AudioProcessingError("No valid audio samples are available.")
    if audio.size < int(0.25 * sample_rate):
        raise AudioProcessingError(
            "Recording is too short to analyze; provide at least 0.25 seconds."
        )

    window = max(1, int(round(window_seconds * sample_rate)))
    hop = max(1, int(round(hop_seconds * sample_rate)))
    if audio.size <= window:
        starts = [0]
    else:
        starts = list(range(0, audio.size - window + 1, hop))
        covered_until = starts[-1] + window
        if audio.size - covered_until >= int(0.5 * sample_rate):
            starts.append(starts[-1] + hop)
    segments = []
    for start in starts:
        chunk = audio[start : start + window]
        if chunk.size < window:
            chunk = np.pad(chunk, (0, window - chunk.size))
        segments.append((start / sample_rate, chunk.astype(np.float32, copy=False)))
    return segments


def apply_audio_effects(
    samples: np.ndarray,
    sample_rate: int,
    *,
    gain_db: float = 0.0,
    echo: float = 0.0,
    reverb: float = 0.0,
    pitch_steps: float = 0.0,
    speed: float = 1.0,
    filter_type: str = "Off",
) -> np.ndarray:
    """Apply non-destructive audio effects and return a new float32 signal."""
    source = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or source.size == 0 or not np.all(np.isfinite(source)):
        raise AudioProcessingError("Valid audio samples and sample rate are required.")
    if not 0.25 <= speed <= 4.0:
        raise AudioProcessingError("Playback speed must be between 0.25x and 4x.")
    if not 0.0 <= echo <= 0.8 or not 0.0 <= reverb <= 0.8:
        raise AudioProcessingError("Echo and reverb amounts must be between 0 and 0.8.")

    output = source.copy()
    if gain_db:
        output *= np.float32(10.0 ** (gain_db / 20.0))

    if echo > 0:
        delay = max(1, int(0.28 * sample_rate))
        echoed = np.zeros(output.size + delay * 3, dtype=np.float32)
        echoed[: output.size] += output
        for repeat in range(1, 4):
            echoed[delay * repeat : delay * repeat + output.size] += (
                output * np.float32(echo**repeat)
            )
        output = echoed

    if reverb > 0:
        ir_length = max(2, int(0.45 * sample_rate))
        impulse = np.zeros(ir_length, dtype=np.float32)
        impulse[0] = 1.0
        for delay_seconds, decay in ((0.031, 0.55), (0.057, 0.38), (0.089, 0.25)):
            delay = int(delay_seconds * sample_rate)
            if delay < ir_length:
                impulse[delay] += np.float32(reverb * decay)
        output = signal.fftconvolve(output, impulse).astype(np.float32)

    if pitch_steps:
        output = librosa.effects.pitch_shift(
            y=output, sr=sample_rate, n_steps=float(pitch_steps)
        ).astype(np.float32)
    if speed != 1.0:
        output = librosa.effects.time_stretch(y=output, rate=float(speed)).astype(
            np.float32
        )

    if filter_type != "Off":
        nyquist = sample_rate / 2.0
        if filter_type == "Low-pass":
            cutoff: float | list[float] = min(4000.0, nyquist * 0.9)
            btype = "lowpass"
        elif filter_type == "High-pass":
            cutoff = min(250.0, nyquist * 0.5)
            btype = "highpass"
        elif filter_type == "Band-pass":
            cutoff = [min(150.0, nyquist * 0.25), min(4000.0, nyquist * 0.9)]
            btype = "bandpass"
        else:
            raise AudioProcessingError(f"Unknown audio filter: {filter_type}")
        if isinstance(cutoff, list) and cutoff[0] >= cutoff[1]:
            raise AudioProcessingError("The sample rate is too low for this filter.")
        sos = signal.butter(4, cutoff, btype=btype, fs=sample_rate, output="sos")
        output = signal.sosfilt(sos, output).astype(np.float32)

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 1.0:
        output /= peak
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def mix_audio_tracks(
    tracks: list[np.ndarray], volumes: list[float]
) -> np.ndarray:
    """Mix equal-sample-rate mono tracks without modifying their inputs."""
    if not tracks or len(tracks) != len(volumes):
        raise AudioProcessingError("Select at least one audio track to mix.")
    arrays = [np.asarray(track, dtype=np.float32).reshape(-1) for track in tracks]
    if any(array.size == 0 or not np.all(np.isfinite(array)) for array in arrays):
        raise AudioProcessingError("Every mix track must contain valid audio.")
    if any(not 0.0 <= volume <= 1.0 for volume in volumes):
        raise AudioProcessingError("Track volumes must be between 0 and 1.")

    mixed = np.zeros(max(array.size for array in arrays), dtype=np.float32)
    for array, volume in zip(arrays, volumes):
        mixed[: array.size] += array * np.float32(volume)
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed /= peak
    return mixed


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono samples to downloadable 16-bit PCM WAV bytes."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or audio.size == 0 or not np.all(np.isfinite(audio)):
        raise AudioProcessingError("Valid audio and sample rate are required for export.")
    output = BytesIO()
    sf.write(output, np.clip(audio, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()
