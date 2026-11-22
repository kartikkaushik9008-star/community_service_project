import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from audio_classification import (
    ClassifierUnavailableError,
    _build_result,
    _top_predictions,
    load_esc50_labels,
    load_esc50_model,
    load_yamnet,
    predict_esc50,
)


class ConstantModel(torch.nn.Module):
    def forward(self, inputs):
        logits = torch.zeros((inputs.shape[0], 50))
        logits[:, 14] = 5.0
        return logits


class AudioClassificationTests(unittest.TestCase):
    def test_label_indices_match_metadata_and_checkpoint_outputs(self):
        labels = load_esc50_labels()
        self.assertEqual(len(labels), 50)
        self.assertEqual(labels[14], "chirping_birds")
        self.assertEqual(labels[2], "pig")

    def test_labels_fall_back_to_packaged_mapping_and_missing_model_is_reported(self):
        labels = load_esc50_labels(metadata_path=Path("missing_esc50.csv"))
        self.assertEqual(labels[14], "chirping_birds")
        with self.assertRaisesRegex(ClassifierUnavailableError, "not found"):
            load_esc50_model(checkpoint_path=Path("missing_model.pth"))

    def test_mismatched_label_count_is_rejected(self):
        with self.assertRaisesRegex(ClassifierUnavailableError, "do not match"):
            _top_predictions(
                np.array([0.7, 0.3, 0.0]), {0: "bird", 1: "pig"}
            )

    def test_aggregate_returns_top_five_and_marks_ambiguous_scores(self):
        labels = {index: f"class_{index}" for index in range(50)}
        scores = np.full(50, 0.01)
        scores[0:2] = [0.35, 0.34]
        result = _build_result([scores, scores], [0.0, 2.5], labels, 0.30, 0.02)
        self.assertEqual(len(result["predictions"]), 5)
        self.assertTrue(result["low_confidence"])
        self.assertEqual(len(result["segments"]), 2)

    def test_original_cnn_predicts_multiple_segments_with_matching_indices(self):
        model = ConstantModel().eval()
        labels = {index: f"class_{index}" for index in range(50)}
        samples = np.ones(7 * 22050, dtype=np.float32) * 0.05
        result = predict_esc50(samples, 22050, model, labels)
        self.assertEqual(result["predictions"][0]["label"], "class_14")
        self.assertEqual(len(result["predictions"]), 5)
        self.assertEqual(len(result["segments"]), 2)

    def test_silence_is_not_given_a_sound_class(self):
        with self.assertRaisesRegex(ClassifierUnavailableError, "No segment"):
            predict_esc50(
                np.zeros(22050, dtype=np.float32),
                22050,
                ConstantModel(),
                {index: f"class_{index}" for index in range(50)},
            )

    def test_yamnet_absent_dependency_is_reported(self):
        with patch("builtins.__import__", side_effect=ImportError("not installed")):
            with self.assertRaisesRegex(ClassifierUnavailableError, "optional"):
                load_yamnet()


if __name__ == "__main__":
    unittest.main()
