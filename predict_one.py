"""Classify one WAV, MP3, OGG, or FLAC recording with the original CNN."""

from pathlib import Path

from audio_classification import (
    ClassifierUnavailableError,
    load_esc50_model,
    predict_esc50,
)
from audio_processing import AudioProcessingError, decode_audio


def main():
    filename = input("Audio path: ").strip()
    if not filename:
        raise SystemExit("Provide a path to an audio file.")
    path = Path(filename).expanduser()
    if not path.is_file():
        raise SystemExit(f"Audio file not found: {path}")

    try:
        decoded = decode_audio(path.read_bytes(), path.name)
        model, labels = load_esc50_model()
        result = predict_esc50(
            decoded.samples, decoded.sample_rate, model, labels
        )
    except (AudioProcessingError, ClassifierUnavailableError) as exc:
        raise SystemExit(f"Could not analyze audio: {exc}") from exc

    if result["low_confidence"]:
        print("Uncertain result; candidates are not confirmed detections:")
    print("Top sound candidates:")
    for rank, prediction in enumerate(result["predictions"], start=1):
        print(
            f"{rank}. {prediction['label']} "
            f"({prediction['confidence']:.1%} model score)"
        )


if __name__ == "__main__":
    main()
