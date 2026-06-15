import sys
sys.path.insert(0, r"i:\cchlink")

import os
import cv2
import numpy as np
from src.preprocess.board_detector import detect_board_corners
from src.preprocess.perspective import warp_board, WARP_PAD
from src.segmentation.grid_splitter import split_board
from src.pipeline import IDX_TO_NAME

os.makedirs(r"i:\cchlink\debug\cells_debug", exist_ok=True)

for test_id in ['004', '007', '015']:
    img_path = rf'i:\cchlink\data\raw\eval\test_{test_id}.png'
    img = cv2.imread(img_path)
    
    corners = detect_board_corners(img)
    board = warp_board(img, corners)
    cropped = board[WARP_PAD:board.shape[0]-WARP_PAD, WARP_PAD:board.shape[1]-WARP_PAD]
    cells = split_board(cropped)
    
    out_dir = rf'i:\cchlink\debug\cells_debug\test_{test_id}'
    os.makedirs(out_dir, exist_ok=True)
    
    for i, cell in enumerate(cells):
        r, c = i // 9, i % 9
        # Save as visible grid
        label = f'r{r}_c{c}_h{cell.shape[0]}x{cell.shape[1]}'
        cv2.imwrite(rf'{out_dir}\{label}.png', cell)
    
    print(f'test_{test_id}: {len(cells)} cells')
    sizes = [(c.shape[0], c.shape[1]) for c in cells]
    h_sizes = [s[0] for s in sizes]
    w_sizes = [s[1] for s in sizes]
    print(f'  Height range: {min(h_sizes)}-{max(h_sizes)}')
    print(f'  Width range: {min(w_sizes)}-{max(w_sizes)}')
    print(f'  Mean sizes: h={np.mean(h_sizes):.1f}, w={np.mean(w_sizes):.1f}')
    print()

    # Also save the cropped board with grid overlay
    vis = cropped.copy()
    from src.segmentation.grid_splitter import _detect_grid_lines
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = _detect_grid_lines(gray, cropped.shape[:2])
    for y in h_lines:
        cv2.line(vis, (0, y), (cropped.shape[1]-1, y), (0, 255, 0), 1)
    for x in v_lines:
        cv2.line(vis, (x, 0), (x, cropped.shape[0]-1), (0, 0, 255), 1)
    cv2.imwrite(rf'{out_dir}\_grid_overlay.png', vis)