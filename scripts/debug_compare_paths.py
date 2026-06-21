import sys

sys.path.insert(0, r"i:\cchlink")

import os

import cv2
import numpy as np

from src.pipeline import _extract_cells_by_homography
from src.preprocess.board_detector import detect_board_corners, detect_grid_homography
from src.preprocess.perspective import WARP_PAD, warp_board
from src.segmentation.grid_splitter import split_board

os.makedirs(r"i:\cchlink\debug\cells\corner", exist_ok=True)
os.makedirs(r"i:\cchlink\debug\cells\homography", exist_ok=True)

for test_id in ["004", "007", "016"]:
    print(f"\n=== test_{test_id} ===")
    img_path = rf"i:\cchlink\data\raw\eval\test_{test_id}.png"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    corners = detect_board_corners(img)
    H = detect_grid_homography(img)

    print(f"  Corners: {'OK' if corners is not None else 'FAIL'}")
    print(f"  Homography: {'OK' if H is not None else 'FAIL'}")

    if corners is not None:
        board = warp_board(img, corners)
        board_cropped = board[
            WARP_PAD : board.shape[0] - WARP_PAD, WARP_PAD : board.shape[1] - WARP_PAD
        ]
        cells = split_board(board_cropped)
        print(f"  Corner cells count: {len(cells)}")
        if cells:
            min_h = min(c.shape[0] for c in cells)
            min_w = min(c.shape[1] for c in cells)
            max_h = max(c.shape[0] for c in cells)
            max_w = max(c.shape[1] for c in cells)
            min_mean = min(float(c.mean()) for c in cells)
            max_mean = max(float(c.mean()) for c in cells)
            print(
                f"  Corner cell sizes range: {min_h}x{min_w} to {max_h}x{max_w}"
            )
            print(f"  Corner cell means range: {min_mean:.1f} to {max_mean:.1f}")
            sample = cells[45] if len(cells) > 45 else cells[len(cells) // 2]
            print(f"  Corner cell[45]: shape={sample.shape}, mean={sample.mean():.1f}")
            os.makedirs(rf"i:\cchlink\debug\cells\corner\test_{test_id}", exist_ok=True)
            for i, cell in enumerate(cells):
                cv2.imwrite(rf"i:\cchlink\debug\cells\corner\test_{test_id}\cell_{i:02d}.png", cell)

    if H is not None:
        cells_4d = _extract_cells_by_homography(img, H)
        means = [float(cells_4d[r, c].mean()) for r in range(10) for c in range(9)]
        zero_count = sum(1 for m in means if m < 1.0)
        print(f"  Homography zero cells: {zero_count}/90")
        print(f"  Homography cell means range: {min(means):.1f} to {max(means):.1f}")
        os.makedirs(rf"i:\cchlink\debug\cells\homography\test_{test_id}", exist_ok=True)
        idx = 0
        for r in range(10):
            for c in range(9):
                cell = cells_4d[r, c].astype(np.uint8)
                cv2.imwrite(
                    rf"i:\cchlink\debug\cells\homography\test_{test_id}\cell_{r}_{c}.png", cell
                )
                idx += 1
    print()
