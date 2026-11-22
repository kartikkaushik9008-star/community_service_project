"""Continuous 5-second microphone analysis using the retained ESC-50 CNN."""

import queue
import threading

import numpy as np
import sounddevice as sd

from audio_classification import (
    ClassifierUnavailableError,
    load_esc50_model,
    predict_esc50,
)
from audio_processing import AudioProcessingError


SAMPLE_RATE = 22050
BLOCK_DURATION = 5.0
BLOCKSIZE = int(SAMPLE_RATE * BLOCK_DURATION)
CONF_THRESHOLD = 0.30
audio_queue = queue.Queue(maxsize=2)


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Microphone status: {status}")
    audio = indata.copy()
    try:
        audio_queue.put_nowait(audio)
    except queue.Full:
        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(audio)
        except (queue.Empty, queue.Full):
            pass


def process_audio(model, idx_to_class):
    print("Listening in 5-second windows... Press Ctrl+C to stop.")
    while True:
        indata = audio_queue.get()
        audio = np.mean(indata, axis=1, dtype=np.float64).astype(np.float32)
        if float(np.sqrt(np.mean(audio**2))) < 0.001:
            continue
        try:
            result = predict_esc50(
                audio, SAMPLE_RATE, model, idx_to_class, CONF_THRESHOLD
            )
        except (AudioProcessingError, ClassifierUnavailableError, ValueError) as exc:
            print(f"Could not analyze microphone audio: {exc}")
            continue
        candidate = result["predictions"][0]
        status = "Uncertain candidate" if result["low_confidence"] else "Candidate"
        print(
            f"{status}: {candidate['label']} "
            f"({candidate['confidence']:.1%} model score)"
        )


def main():
    try:
        model, idx_to_class = load_esc50_model()
    except ClassifierUnavailableError as exc:
        raise SystemExit(f"Could not start microphone analysis: {exc}") from exc

    threading.Thread(
        target=process_audio, args=(model, idx_to_class), daemon=True
    ).start()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCKSIZE,
            callback=audio_callback,
            dtype="float32",
        ):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped microphone analysis.")
    except Exception as exc:
        raise SystemExit(
            f"Could not open the microphone at {SAMPLE_RATE} Hz: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
