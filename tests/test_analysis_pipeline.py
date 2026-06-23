from types import SimpleNamespace

import numpy as np

from src.fen.fen_builder import EMPTY_IDX
from src.pipeline import STANDARD_INITIAL_INDICES, Pipeline


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


def test_empty_calibration_rolls_back_low_margin_piece_predictions():
    pipeline = Pipeline.__new__(Pipeline)
    predictions = [14] * 90
    confidences = [0.99] * 90
    probabilities = np.zeros((90, 15), dtype=np.float32)
    probabilities[:, 14] = 0.99
    predictions[10] = 4
    confidences[10] = 0.45
    probabilities[10, 4] = 0.45
    probabilities[10, 14] = 0.30
    predictions[11] = 3
    confidences[11] = 0.80
    probabilities[11, 3] = 0.80
    probabilities[11, 14] = 0.15

    calibrated, calibrated_confidences, corrections = pipeline._calibrate_empty_predictions(
        predictions, confidences, probabilities
    )

    assert calibrated[10] == 14
    assert calibrated_confidences[10] == probabilities[10, 14]
    assert calibrated[11] == 3
    assert len(corrections) == 1
    assert corrections[0].reason == "低边际空位误报回退"


def test_initial_position_prior_fills_near_complete_initial_board():
    pipeline = Pipeline.__new__(Pipeline)
    predictions = STANDARD_INITIAL_INDICES.copy()
    confidences = [0.80] * 90
    missing_index = 0
    predictions[missing_index] = EMPTY_IDX
    confidences[missing_index] = 0.60
    probabilities = np.full((90, 15), 0.001, dtype=np.float32)
    for index, target in enumerate(STANDARD_INITIAL_INDICES):
        probabilities[index, target] = 0.99
    probabilities[missing_index, EMPTY_IDX] = 0.60
    probabilities[missing_index, STANDARD_INITIAL_INDICES[missing_index]] = 0.50

    corrected, corrected_confidences, corrections = pipeline._apply_initial_position_prior(
        predictions, confidences, probabilities
    )

    assert corrected == STANDARD_INITIAL_INDICES
    assert corrected_confidences[missing_index] == probabilities[
        missing_index, STANDARD_INITIAL_INDICES[missing_index]
    ]
    assert len(corrections) == 1
    assert corrections[0].reason == "近完整初始局模板补全"


def test_initial_position_prior_rejects_when_current_board_is_much_more_likely():
    pipeline = Pipeline.__new__(Pipeline)
    predictions = STANDARD_INITIAL_INDICES.copy()
    confidences = [0.80] * 90
    occupied_indices = [
        index for index, target in enumerate(STANDARD_INITIAL_INDICES) if target != EMPTY_IDX
    ]
    empty_indices = [
        index for index, target in enumerate(STANDARD_INITIAL_INDICES) if target == EMPTY_IDX
    ]
    wrong_occupied_indices = occupied_indices[:22]
    false_positive_indices = empty_indices[:58]
    for index in wrong_occupied_indices:
        predictions[index] = 4
    for index in false_positive_indices:
        predictions[index] = 4
    probabilities = np.full((90, 15), 0.001, dtype=np.float32)
    for index, target in enumerate(STANDARD_INITIAL_INDICES):
        probabilities[index, target] = 0.99
    for index in wrong_occupied_indices:
        probabilities[index, STANDARD_INITIAL_INDICES[index]] = 0.05
        probabilities[index, predictions[index]] = 0.99
    for index in false_positive_indices:
        probabilities[index, EMPTY_IDX] = 0.71
        probabilities[index, predictions[index]] = 0.99

    corrected, corrected_confidences, corrections = pipeline._apply_initial_position_prior(
        predictions, confidences, probabilities
    )

    assert corrected == predictions
    assert corrected_confidences == confidences
    assert corrections == []


def test_visual_initial_position_prior_fills_red_top_low_recall_board():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._red_pixel_fraction = lambda board: 0.30
    pipeline._visual_initial_occupied_hits = lambda board, rows, cols: 25
    predictions = [EMPTY_IDX] * 90
    confidences = [0.60] * 90
    probabilities = np.full((90, 15), 0.001, dtype=np.float32)
    for index, target in enumerate(STANDARD_INITIAL_INDICES):
        probabilities[index, target] = 0.40

    corrected, corrected_confidences, corrections = pipeline._apply_visual_initial_position_prior(
        predictions,
        confidences,
        probabilities,
        np.zeros((100, 100, 3), dtype=np.uint8),
        list(range(10)),
        list(range(9)),
        "red_top",
    )

    assert corrected == STANDARD_INITIAL_INDICES
    assert corrected_confidences[0] == probabilities[0, STANDARD_INITIAL_INDICES[0]]
    assert corrections
    assert corrections[0].reason == "瑙嗚鍒濆灞€鍗犱綅琛ュ叏"


def test_visual_initial_position_prior_rejects_without_red_piece_evidence():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._red_pixel_fraction = lambda board: 0.10
    pipeline._visual_initial_occupied_hits = lambda board, rows, cols: 30
    predictions = [EMPTY_IDX] * 90
    confidences = [0.60] * 90

    corrected, corrected_confidences, corrections = pipeline._apply_visual_initial_position_prior(
        predictions,
        confidences,
        None,
        np.zeros((100, 100, 3), dtype=np.uint8),
        list(range(10)),
        list(range(9)),
        "red_top",
    )

    assert corrected == predictions
    assert corrected_confidences == confidences
    assert corrections == []


def test_visual_initial_position_prior_fills_high_contrast_initial_board():
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._red_pixel_fraction = lambda board: 0.0
    pipeline._mean_saturation = lambda board: 45.0
    pipeline._visual_initial_occupied_hits = lambda board, rows, cols: 31
    predictions = [EMPTY_IDX] * 90
    confidences = [0.60] * 90

    corrected, corrected_confidences, corrections = pipeline._apply_visual_initial_position_prior(
        predictions,
        confidences,
        None,
        np.zeros((100, 100, 3), dtype=np.uint8),
        list(range(10)),
        list(range(9)),
        "red_bottom",
    )

    assert corrected == STANDARD_INITIAL_INDICES
    assert corrected_confidences == confidences
    assert corrections


def test_static_position_prior_replaces_illegal_pawn_with_probable_elephant():
    pipeline = Pipeline.__new__(Pipeline)
    predictions = [EMPTY_IDX] * 90
    confidences = [0.99] * 90
    index = 2 * 9 + 8
    predictions[index] = 13
    confidences[index] = 0.62
    probabilities = np.full((90, 15), 0.001, dtype=np.float32)
    probabilities[:, EMPTY_IDX] = 0.99
    probabilities[index, 13] = 0.62
    probabilities[index, 9] = 0.24

    corrected, corrected_confidences, corrections = (
        pipeline._apply_static_position_probability_prior(
            predictions, confidences, probabilities
        )
    )

    assert corrected[index] == 9
    assert corrected_confidences[index] == probabilities[index, 9]
    assert len(corrections) == 1
    assert corrections[0].reason == "非法兵卒位置概率先验修正"


def test_static_position_prior_replaces_weak_pawn_on_elephant_point():
    pipeline = Pipeline.__new__(Pipeline)
    predictions = [EMPTY_IDX] * 90
    confidences = [0.99] * 90
    index = 4 * 9 + 6
    predictions[index] = 13
    confidences[index] = 0.43
    probabilities = np.full((90, 15), 0.001, dtype=np.float32)
    probabilities[:, EMPTY_IDX] = 0.99
    probabilities[index, 13] = 0.43
    probabilities[index, 9] = 0.31

    corrected, _, corrections = pipeline._apply_static_position_probability_prior(
        predictions, confidences, probabilities
    )

    assert corrected[index] == 9
    assert corrections


def test_sparse_endgame_identity_prior_restores_missing_red_king_and_horse():
    pipeline = Pipeline.__new__(Pipeline)
    predictions = [EMPTY_IDX] * 90
    confidences = [0.99] * 90
    cannon_index = 7 * 9 + 4
    king_index = 9 * 9 + 4
    predictions[4] = 7
    predictions[5] = 8
    predictions[cannon_index] = 5
    confidences[cannon_index] = 0.42
    predictions[king_index] = 6
    confidences[king_index] = 0.55

    corrected, corrected_confidences, corrections = (
        pipeline._apply_sparse_endgame_identity_prior(predictions, confidences)
    )

    assert corrected[king_index] == 0
    assert corrected[cannon_index] == 4
    assert corrected_confidences == confidences
    assert [correction.reason for correction in corrections] == [
        "稀疏残局缺少红帅位置先验",
        "稀疏残局九宫红炮马形近先验",
    ]


def test_sparse_endgame_identity_prior_rejects_dense_boards():
    pipeline = Pipeline.__new__(Pipeline)
    predictions = [EMPTY_IDX] * 90
    confidences = [0.40] * 90
    for index in range(7):
        predictions[index] = 5
    predictions[9 * 9 + 4] = 6

    corrected, _, corrections = pipeline._apply_sparse_endgame_identity_prior(
        predictions, confidences
    )

    assert corrected == predictions
    assert corrections == []
