import numpy as np
import torch

from src.recognition.predictor import PiecePredictor


class _FakeModel(torch.nn.Module):
    def __init__(self, class_index=14):
        super().__init__()
        self.calls = 0
        self.class_index = class_index

    def forward(self, inputs):
        self.calls += 1
        outputs = torch.zeros((len(inputs), 15))
        outputs[:, self.class_index] = 10
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


def test_predict_batch_probabilities_averages_ensemble_models():
    predictor = PiecePredictor.__new__(PiecePredictor)
    predictor.device = "cpu"
    predictor.models = [_FakeModel(class_index=3), _FakeModel(class_index=14)]
    predictor.model = predictor.models[0]
    predictor.model_weights = [0.25, 0.75]

    probabilities = predictor.predict_batch_probabilities(
        [np.zeros((64, 64, 3), dtype=np.uint8)]
    )

    assert probabilities[0, 14] > probabilities[0, 3]
    assert all(model.calls == 1 for model in predictor.models)
