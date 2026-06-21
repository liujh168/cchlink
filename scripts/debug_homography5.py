import sys

sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np

from src.preprocess.board_detector import _compute_grid_homography
from src.preprocess.perspective import BOARD_HEIGHT, BOARD_WIDTH

test_id = "004"
img_path = rf"i:\cchlink\data\raw\eval\test_{test_id}.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

for canny_low, canny_high, hough_thresh in [(40, 120, 0.15), (40, 120, 0.12), (50, 150, 0.10)]:
    H = _compute_grid_homography(gray, h, w, canny_low, canny_high, hough_thresh)
    if H is not None:
        print(f"Found H with canny={canny_low},{canny_high} thresh={hough_thresh}")
        print(f"H matrix:\n{H}")

        H_inv = np.linalg.inv(H)

        # Test mapping for cell (0,0) and (4,4) and (8,9)
        for r, c in [(0, 0), (4, 4), (8, 9), (0, 4), (4, 0), (8, 0), (0, 9)]:
            row_px = r * (BOARD_HEIGHT - 1) / 10.0
            col_px = c * (BOARD_WIDTH - 1) / 9.0
            cell_w = (BOARD_WIDTH - 1) / 9.0
            cell_h = (BOARD_HEIGHT - 1) / 10.0
            half_w = cell_w * 0.4
            half_h = cell_h * 0.4

            dst_center = np.array([[col_px], [row_px], [1.0]])
            src_center = H_inv @ dst_center
            src_center = src_center[:2] / src_center[2]
            sx, sy = src_center[0, 0], src_center[1, 0]

            # Also check corners
            corners = np.array(
                [
                    [col_px - half_w, row_px - half_h, 1.0],
                    [col_px + half_w, row_px - half_h, 1.0],
                    [col_px + half_w, row_px + half_h, 1.0],
                    [col_px - half_w, row_px + half_h, 1.0],
                ]
            ).T
            src_corners = H_inv @ corners
            src_corners = src_corners[:2] / src_corners[2]

            in_bounds = all(0 <= x < w and 0 <= y < h for x, y in src_corners.T)

            status = "OK" if in_bounds else "OUT"
            print(
                f"  cell({r},{c}): px=({col_px:.0f},{row_px:.0f}) -> "
                f"src=({sx:.0f},{sy:.0f}) {status}"
            )

        # Check what cv2.findHomography actually produces
        # The H maps src -> dst (image -> board pixel)
        # H_inv maps board pixel -> image

        # Let me test one known intersection point
        break
