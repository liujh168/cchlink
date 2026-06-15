from types import SimpleNamespace

import numpy as np

from src.pipeline import Pipeline


class _Predictor:
    def __init__(self, predictions, confidences):
        self.predictions = predictions
        self.confidences = confidences

    def predict_batch(self, cells):
        return self.predictions, self.confidences


def _pipeline(predictions, confidences):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.predictor = _Predictor(predictions, confidences)
    pipeline.apply_rules = True
    detection = SimpleNamespace(confidence=0.9, corners=np.zeros((4, 2), dtype=np.float32))
    grid = SimpleNamespace(
        confidence=0.8,
        row_positions=list(range(10)),
        col_positions=list(range(9)),
    )
    board = np.zeros((10, 9, 3), dtype=np.uint8)
    cells = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(90)]
    pipeline._extract = lambda image: (detection, grid, board, cells)
    return pipeline


def test_analyze_normalizes_direction_and_records_corrections():
    predictions = [14] * 90
    confidences = [0.99] * 90
    predictions[4] = 0
    predictions[85] = 7
    predictions[0] = 0
    confidences[0] = 0.1
    pipeline = _pipeline(predictions, confidences)

    result = pipeline.analyze(np.zeros((10, 9, 3), dtype=np.uint8))

    assert result.orientation == "red_top"
    assert result.raw_fen != result.fen
    assert result.corrections
    assert len(result.cells) == 90
    assert result.cells[89].raw_prediction != result.cells[89].prediction
    correction = result.corrections[0]
    corrected_cell = result.cells[correction.row * 9 + correction.col]
    assert corrected_cell.raw_prediction == correction.before
    assert corrected_cell.prediction == correction.after


def test_run_and_verbose_remain_compatible():
    predictions = [14] * 90
    predictions[4] = 7
    predictions[85] = 0
    pipeline = _pipeline(predictions, [0.99] * 90)
    image = np.zeros((10, 9, 3), dtype=np.uint8)

    assert isinstance(pipeline.run(image), str)
    verbose = pipeline.run_verbose(image)
    assert verbose["fen"] == pipeline.run(image)
    assert len(verbose["grid"]) == 10
    assert "warnings" in verbose


def test_analyze_writes_visualization_and_debug_independently(monkeypatch):
    predictions = [14] * 90
    predictions[4] = 7
    predictions[85] = 0
    pipeline = _pipeline(predictions, [0.99] * 90)
    calls = []
    monkeypatch.setattr("src.pipeline.save_visualizations", lambda *args: calls.append("visual"))
    monkeypatch.setattr("src.pipeline.save_debug_artifacts", lambda *args: calls.append("debug"))

    pipeline.analyze(
        np.zeros((10, 9, 3), dtype=np.uint8),
        visualize_dir="visual",
        debug_dir="debug",
    )

    assert calls == ["visual", "debug"]
