import sys

sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np

from src.preprocess.board_detector import detect_board_corners
from src.preprocess.perspective import WARP_PAD, warp_board
from src.segmentation.grid_splitter import _detect_grid_lines

image_path = r"i:\cchlink\data\raw\initial_00.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

corners = detect_board_corners(image)
print(f"Corners: {corners}")

board = warp_board(image, corners)
print(f"Warped shape: {board.shape}")

board_cropped = board[WARP_PAD : board.shape[0] - WARP_PAD, WARP_PAD : board.shape[1] - WARP_PAD]
print(f"Cropped shape: {board_cropped.shape}")

gray = cv2.cvtColor(board_cropped, cv2.COLOR_RGB2GRAY)
h_lines, v_lines = _detect_grid_lines(gray, board_cropped.shape[:2])
print(f"H lines ({len(h_lines)}): {h_lines}")
print(f"V lines ({len(v_lines)}): {v_lines}")

edge_h = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
profile_h = edge_h.mean(axis=1)
profile_h = np.convolve(profile_h, np.ones(5) / 5, mode="same")
print(f"H profile (first 100): {profile_h[:100].round(1).tolist()}")

edge_v = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
profile_v = edge_v.mean(axis=0)
profile_v = np.convolve(profile_v, np.ones(5) / 5, mode="same")
print(f"V profile (first 100): {profile_v[:100].round(1).tolist()}")

vis = board_cropped.copy()
for y in h_lines:
    cv2.line(vis, (0, y), (board_cropped.shape[1], y), (0, 255, 0), 1)
for x in v_lines:
    cv2.line(vis, (x, 0), (x, board_cropped.shape[0]), (255, 0, 0), 1)
cv2.imwrite(r"i:\cchlink\data\raw\diag_grid.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print("Saved diag_grid.jpg")
