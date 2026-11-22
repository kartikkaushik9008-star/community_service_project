import sounddevice as sd
import numpy as np

SAMPLE_RATE = 22050
BLOCKSIZE = 22050  

def audio_callback(indata, frames, time, status):
    if status:
        print("Status:", status)
    level = float(np.linalg.norm(indata) / len(indata))
    print(f"🔊 Level: {level:.4f}")

print("Listing input devices:")
print(sd.query_devices())

print("🎤 Listening on default input device... Ctrl+C to stop")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=BLOCKSIZE,
    callback=audio_callback,
    dtype='float32',
    device=None  
):
    sd.sleep(1000000)
