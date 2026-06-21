import sys

sys.path.insert(0, r"i:\cchlink")
import importlib

import cv2
import numpy as np

import src.preprocess.board_detector as bd

importlib.reload(bd)

from src.pipeline import _extract_cells_by_homography

for test_id in ["000", "001", "002", "004", "006"]:
    print(f"\n=== test_{test_id} ===")
    img = cv2.imread(rf"i:\cchlink\data\raw\eval\test_{test_id}.png")
    h, w = img.shape[:2]
    H = bd.detect_grid_homography(img)
    print(f"H detected: {H is not None}")
    if H is None:
        continue

    cells = _extract_cells_by_homography(img, H)
    print(f"cells shape: {cells.shape}")

    cell_means = []
    for r in range(10):
        for c in range(9):
            cell = cells[r, c]
            cell_means.append(float(cell.mean()))

    min_mean = min(cell_means)
    max_mean = max(cell_means)
    avg_mean = np.mean(cell_means)
    print(f"cell mean: min={min_mean:.1f} max={max_mean:.1f} avg={avg_mean:.1f}")
    print(f"sample cell (5,4) mean: {cells[5, 4].mean():.1f}")
    print(f"sample cell (0,0) mean: {cells[0, 0].mean():.1f}")
    print(f"sample cell (9,8) mean: {cells[9, 8].mean():.1f}")
