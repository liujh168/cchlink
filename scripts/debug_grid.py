import sys

sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np

from src.preprocess.board_detector import detect_board_corners
from src.preprocess.perspective import warp_board
from src.segmentation.grid_splitter import _find_grid_bounds, _split_adaptive

image_path = r"i:\cchlink\data\raw\initial_00.jpg"
image = cv2.imread(image_path)
corners = detect_board_corners(image)
board = warp_board(image, corners)
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

print(f"warped shape: {board.shape}")
top, bottom, left, right = _find_grid_bounds(gray)
print(f"grid bounds: top={top}, bottom={bottom}, left={left}, right={right}")
print(f"grid size: {bottom - top} x {right - left} (expected 500 x 450)")

h_lines, v_lines = _split_adaptive(board, top, bottom, left, right)
print(f"h_lines: {h_lines}")
print(f"v_lines: {v_lines}")

# Check some cells
for r in [0, 2, 7, 9]:
    for c in [0, 4, 8]:
        y1, y2 = h_lines[r], h_lines[r + 1]
        x1, x2 = v_lines[c], v_lines[c + 1]
        cx1, cy1 = x1 + 5, y1 + 5
        cx2, cy2 = x2 - 5, y2 - 5
        cell = board[cy1:cy2, cx1:cx2]
        cv2.imwrite(rf"i:\cchlink\data\raw\cell_v2_r{r}_c{c}.jpg", cell)
        print(f"cell ({r},{c}): [{cx1},{cy1}] - [{cx2},{cy2}] shape={cell.shape}")

# Also check the Sobel profiles
edge_h = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
h_profile = edge_h.mean(axis=1)
edge_v = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
v_profile = edge_v.mean(axis=0)

top_end = gray.shape[0] // 4
bottom_start = 3 * gray.shape[0] // 4
left_end = gray.shape[1] // 4
right_start = 3 * gray.shape[1] // 4
top_peak = np.argmax(h_profile[:top_end])
bottom_peak = bottom_start + np.argmax(h_profile[bottom_start:])
left_peak = np.argmax(v_profile[:left_end])
right_peak = right_start + np.argmax(v_profile[right_start:])

print(f"\nTop edge region (0-{top_end}): peaks at {top_peak}")
print(f"Bot edge region ({bottom_start}-{gray.shape[0]}): peaks at {bottom_peak}")
print(f"Left edge region (0-{left_end}): peaks at {left_peak}")
print(f"Right edge region ({right_start}-{gray.shape[1]}): peaks at {right_peak}")

# Save the profile data
np.save(r"i:\cchlink\data\raw\h_profile.npy", h_profile)
np.save(r"i:\cchlink\data\raw\v_profile.npy", v_profile)
print("Saved h_profile.npy and v_profile.npy")
