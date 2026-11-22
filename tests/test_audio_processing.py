import io
import unittest

import numpy as np
import soundfile as sf

from audio_processing import (
    AudioProcessingError,
    apply_audio_effects,
    decode_audio,
    encode_wav,
    mix_audio_tracks,
    segment_audio,
)


def wav_bytes(samples, sample_rate):
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="WAV")
    return output.getvalue()


class AudioProcessingTests(unittest.TestCase):
    def test_decode_resamples_stereo_to_mono(self):
        rate = 44100
        samples = np.column_stack(
            [np.full(rate, 0.2, dtype=np.float32), np.full(rate, -0.2, dtype=np.float32)]
        )
        decoded = decode_audio(wav_bytes(samples, rate), "test.wav")
        self.assertEqual(decoded.sample_rate, 22050)
        self.assertEqual(decoded.source_sample_rate, rate)
        self.assertEqual(decoded.samples.ndim, 1)
        self.assertEqual(decoded.samples.size, 22050)
        self.assertAlmostEqual(float(np.max(np.abs(decoded.samples))), 0.0, places=4)

    def test_invalid_and_empty_uploads_have_clear_errors(self):
        for content, name in ((b"", "empty.wav"), (b"not audio", "bad.wav")):
            with self.subTest(name=name), self.assertRaises(AudioProcessingError):
                decode_audio(content, name)
        with self.assertRaisesRegex(AudioProcessingError, "Unsupported"):
            decode_audio(b"data", "recording.txt")

    def test_segments_include_padding_for_short_valid_recording(self):
        samples = np.ones(22050, dtype=np.float32)
        segments = segment_audio(samples, 22050)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][1].size, 5 * 22050)
        self.assertEqual(float(segments[0][1][-1]), 0.0)
        with self.assertRaisesRegex(AudioProcessingError, "0.25 seconds"):
            segment_audio(np.ones(1000, dtype=np.float32), 22050)

    def test_effects_are_non_destructive_and_mixdown_exports_wav(self):
        rate = 22050
        original = np.full(rate, 0.1, dtype=np.float32)
        preserved = original.copy()
        processed = apply_audio_effects(
            original, rate, gain_db=6, echo=0.2, reverb=0.2, filter_type="Low-pass"
        )
        self.assertTrue(np.array_equal(original, preserved))
        self.assertGreater(processed.size, original.size)
        self.assertTrue(np.all(np.isfinite(processed)))

        mixed = mix_audio_tracks([original, original], [0.8, 0.8])
        self.assertAlmostEqual(float(np.max(np.abs(mixed))), 0.16, places=5)
        louder_mix = mix_audio_tracks(
            [np.full(rate, 0.8, dtype=np.float32)] * 2, [1.0, 1.0]
        )
        self.assertAlmostEqual(float(np.max(np.abs(louder_mix))), 1.0, places=5)
        self.assertTrue(encode_wav(mixed, rate).startswith(b"RIFF"))

    def test_pitch_and_speed_changes_return_valid_audio(self):
        rate = 22050
        time = np.arange(rate, dtype=np.float32) / rate
        original = np.sin(2 * np.pi * 440 * time).astype(np.float32)
        pitched = apply_audio_effects(original, rate, pitch_steps=2)
        slowed = apply_audio_effects(original, rate, speed=0.8)
        self.assertEqual(pitched.size, original.size)
        self.assertGreater(slowed.size, original.size)
        self.assertTrue(np.all(np.isfinite(pitched)))


if __name__ == "__main__":
    unittest.main()
