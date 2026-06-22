import cv2
import numpy as np

from src.geometry import ideal_grid_positions
from src.preprocess.board_detector import _hough_intersection, _score_candidate, detect_board


def test_candidate_scoring_finds_grid_board():
    image = np.full((620, 620, 3), 40, dtype=np.uint8)
    board = np.full((500, 450, 3), 230, dtype=np.uint8)
    rows, cols = ideal_grid_positions(450, 500)
    for y in rows:
        cv2.line(board, (0, y), (449, y), (20, 20, 20), 2)
    for x in cols:
        cv2.line(board, (x, 0), (x, 499), (20, 20, 20), 2)
    image[60:560, 85:535] = board

    detection = detect_board(image)

    assert detection is not None
    assert detection.grid_confidence > 0.5


def test_hough_intersection_preserves_xy_order():
    point = _hough_intersection(12, 0, 34, np.pi / 2)
    assert np.allclose(point, (12, 34))


def test_candidate_scoring_penalizes_tiny_local_grid_crop(monkeypatch):
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    full_board = np.array([[0, 0], [900, 0], [900, 999], [0, 999]], dtype=np.float32)
    local_grid = np.array([[600, 200], [950, 200], [950, 500], [600, 500]], dtype=np.float32)

    scores = iter([0.30, 0.46])
    monkeypatch.setattr(
        "src.preprocess.board_detector.detect_grid",
        lambda image: type("Grid", (), {"confidence": next(scores)})(),
    )

    full_detection = _score_candidate(image, full_board)
    local_detection = _score_candidate(image, local_grid)

    assert full_detection.confidence > local_detection.confidence
