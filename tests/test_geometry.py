import cv2
import numpy as np

from src.geometry import COLS, ROWS, extract_intersection_patches, ideal_grid_positions
from src.segmentation.grid_splitter import _regularize_intersections, detect_grid, split_board


def _synthetic_grid(width=450, height=500):
    image = np.full((height, width, 3), 230, dtype=np.uint8)
    rows, cols = ideal_grid_positions(width, height)
    for y in rows:
        cv2.line(image, (0, y), (width - 1, y), (20, 20, 20), 2)
    for x in cols:
        cv2.line(image, (x, 0), (x, height - 1), (20, 20, 20), 2)
    return image, rows, cols


def test_grid_detection_and_intersection_patches():
    image, expected_rows, expected_cols = _synthetic_grid()
    detection = detect_grid(image)

    assert len(detection.row_positions) == ROWS
    assert len(detection.col_positions) == COLS
    assert max(abs(a - b) for a, b in zip(detection.row_positions, expected_rows)) <= 2
    assert max(abs(a - b) for a, b in zip(detection.col_positions, expected_cols)) <= 2
    assert detection.confidence > 0.5
    assert len(split_board(image)) == ROWS * COLS


def test_edge_intersections_are_padded():
    image, rows, cols = _synthetic_grid()
    patches = extract_intersection_patches(image, rows, cols)

    assert len(patches) == ROWS * COLS
    assert all(patch.shape == (64, 64, 3) for patch in patches)


def test_blank_image_has_low_grid_confidence():
    blank = np.full((500, 450, 3), 128, dtype=np.uint8)
    assert detect_grid(blank).confidence < 0.01


def test_intersection_regularization_ignores_local_outliers():
    ideal = [25, 75, 125, 175, 225]
    snapped = [36, 78, 127, 176, 236]

    regularized = _regularize_intersections(snapped, ideal)

    assert regularized == [28, 78, 128, 178, 228]
