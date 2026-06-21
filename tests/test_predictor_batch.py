import numpy as np
import torch

from src.recognition.predictor import PiecePredictor


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        outputs = torch.zeros((len(inputs), 15))
        outputs[:, 14] = 10
        return outputs


def test_predict_batch_uses_one_forward_pass():
    predictor = PiecePredictor.__new__(PiecePredictor)
    predictor.device = "cpu"
    predictor.model = _FakeModel()

    predictions, confidences = predictor.predict_batch(
        [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(90)]
    )

    assert predictions == [14] * 90
    assert len(confidences) == 90
    assert predictor.model.calls == 1


def test_predict_batch_probabilities_returns_full_distribution():
    predictor = PiecePredictor.__new__(PiecePredictor)
    predictor.device = "cpu"
    predictor.model = _FakeModel()

    probabilities = predictor.predict_batch_probabilities(
        [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    )

    assert probabilities.shape == (3, 15)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
