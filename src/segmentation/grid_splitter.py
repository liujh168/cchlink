import cv2
import numpy as np
from src.preprocess.perspective import BOARD_WIDTH, BOARD_HEIGHT, WARP_PAD


COLS = 9
ROWS = 10
CELL_SIZE = 50
LINE_SEARCH_RANGE = 6


def _refine_grid_lines(gray: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = gray.shape

    edge_h = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    edge_v = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))

    h_profile = edge_h.mean(axis=1)
    v_profile = edge_v.mean(axis=0)

    h_lines = []
    for r in range(ROWS + 1):
        exp_y = WARP_PAD + r * CELL_SIZE
        y0 = max(0, exp_y - LINE_SEARCH_RANGE)
        y1 = min(h - 1, exp_y + LINE_SEARCH_RANGE)
        best_y = int(np.argmax(h_profile[y0:y1 + 1])) + y0
        h_lines.append(best_y)

    v_lines = []
    for c in range(COLS + 1):
        exp_x = WARP_PAD + c * CELL_SIZE
        x0 = max(0, exp_x - LINE_SEARCH_RANGE)
        x1 = min(w - 1, exp_x + LINE_SEARCH_RANGE)
        best_x = int(np.argmax(v_profile[x0:x1 + 1])) + x0
        v_lines.append(best_x)

    return h_lines, v_lines


def split_board(board_image: np.ndarray) -> list[np.ndarray]:
    cells = []
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = _refine_grid_lines(gray)

    for row in range(ROWS):
        y1 = h_lines[row]
        y2 = h_lines[row + 1]
        for col in range(COLS):
            x1 = v_lines[col]
            x2 = v_lines[col + 1]
            cell = board_image[y1:y2, x1:x2]
            cells.append(cell)

    return cells


def split_board_with_positions(board_image: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    cells = []
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = _refine_grid_lines(gray)

    for row in range(ROWS):
        y1 = h_lines[row]
        y2 = h_lines[row + 1]
        for col in range(COLS):
            x1 = v_lines[col]
            x2 = v_lines[col + 1]
            cell = board_image[y1:y2, x1:x2]
            cells.append((row, col, cell))

    return cells