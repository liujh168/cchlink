import sys
sys.path.insert(0, r'i:\cchlink')
import importlib

import cv2, numpy as np
import src.preprocess.board_detector as bd
importlib.reload(bd)

from src.preprocess.perspective import warp_board, WARP_PAD
from src.segmentation.grid_splitter import _detect_grid_lines

img = cv2.imread(r'i:\cchlink\data\raw\eval\test_006.png')
h_img, w_img = img.shape[:2]
print(f'image: {w_img}x{h_img}')

corners = bd.detect_board_corners(img)
print(f'corners detected: {corners is not None}')
if corners is not None:
    print(f'corners: {corners.tolist()}')
    board = warp_board(img, corners)
    pad = WARP_PAD
    board_cropped = board[pad:board.shape[0]-pad, pad:board.shape[1]-pad]
    h, w = board_cropped.shape[:2]
    print(f'cropped board: {w}x{h}')

    gray = cv2.cvtColor(board_cropped, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = _detect_grid_lines(gray, (h, w))
    
    h_spacing = (h_lines[-1] - h_lines[0]) / 9.0
    h_expected = [h_lines[0] + i * h_spacing for i in range(10)]
    h_devs = [abs(h_lines[i] - h_expected[i]) for i in range(min(10, len(h_lines)))]
    
    v_spacing = (v_lines[-1] - v_lines[0]) / 8.0
    v_expected = [v_lines[0] + i * v_spacing for i in range(9)]
    v_devs = [abs(v_lines[i] - v_expected[i]) for i in range(min(9, len(v_lines)))]
    
    print(f'h_lines ({len(h_lines)}): {h_lines}')
    print(f'  expected: {[int(e) for e in h_expected]}')
    print(f'  devs: {[int(d) for d in h_devs]}  max={max(h_devs):.1f}')
    print(f'v_lines ({len(v_lines)}): {v_lines}')
    print(f'  expected: {[int(e) for e in v_expected]}')
    print(f'  devs: {[int(d) for d in v_devs]}  max={max(v_devs):.1f}')