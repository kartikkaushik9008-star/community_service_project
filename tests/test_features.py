import tempfile
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf

from features import extract_mfcc


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_vector_matches_checkpoint_input_width(self):
        sample_rate = 22050
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = 0.1 * np.sin(2 * np.pi * 440 * time)
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            sf.write(audio_path, waveform, sample_rate)
            features = extract_mfcc(audio_path)
        self.assertEqual(features.shape, (128,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))


if __name__ == "__main__":
    unittest.main()
