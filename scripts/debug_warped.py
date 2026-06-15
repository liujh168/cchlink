import sys
sys.path.insert(0, r"i:\cchlink")

import os
import cv2
import numpy as np
from src.preprocess.board_detector import detect_board_corners
from src.preprocess.perspective import warp_board, WARP_PAD
from src.segmentation.grid_splitter import _detect_grid_lines

os.makedirs(r"i:\cchlink\debug\warped", exist_ok=True)

for test_id in ['004', '007', '015', '016']:
    print(f'\n=== test_{test_id} ===')
    img_path = rf'i:\cchlink\data\raw\eval\test_{test_id}.png'
    img = cv2.imread(img_path)
    
    corners = detect_board_corners(img)
    if corners is None:
        print('  Corner detection failed')
        continue
    
    board = warp_board(img, corners)
    h_full, w_full = board.shape[:2]
    print(f'  Warped board: {board.shape}')
    
    cv2.imwrite(rf'i:\cchlink\debug\warped\test_{test_id}_board.png', board)
    
    cropped = board[WARP_PAD:h_full-WARP_PAD, WARP_PAD:w_full-WARP_PAD]
    h, w = cropped.shape[:2]
    print(f'  Cropped: {cropped.shape}')
    
    cv2.imwrite(rf'i:\cchlink\debug\warped\test_{test_id}_cropped.png', cropped)
    
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = _detect_grid_lines(gray, cropped.shape[:2])
    print(f'  h_lines ({len(h_lines)}): {h_lines}')
    print(f'  v_lines ({len(v_lines)}): {v_lines}')
    
    vis = cropped.copy()
    for y in h_lines:
        cv2.line(vis, (0, y), (w-1, y), (0, 255, 0), 1)
    for x in v_lines:
        cv2.line(vis, (x, 0), (x, h-1), (0, 0, 255), 1)
    cv2.imwrite(rf'i:\cchlink\debug\warped\test_{test_id}_grid.png', vis)
    
    ideal_h = [int(i * h / 10.0) for i in range(11)]
    ideal_v = [int(i * w / 9.0) for i in range(10)]
    print(f'  ideal h: {ideal_h}')
    print(f'  ideal v: {ideal_v}')
    print(f'  h offset: {[h_lines[i]-ideal_h[i] for i in range(min(len(h_lines),11))]}')
    print(f'  v offset: {[v_lines[i]-ideal_v[i] for i in range(min(len(v_lines),10))]}')
    print()