import cv2
import numpy as np
from src.preprocess.perspective import BOARD_WIDTH, BOARD_HEIGHT


COLS = 9
ROWS = 10
CELL_SIZE = 50
CROP_RATIO = 0.8


def split_board(board_image: np.ndarray) -> list[np.ndarray]:
    cells = []
    margin = int(CELL_SIZE * (1 - CROP_RATIO) / 2)

    for row in range(ROWS):
        for col in range(COLS):
            x1 = col * CELL_SIZE + margin
            y1 = row * CELL_SIZE + margin
            x2 = x1 + CELL_SIZE - 2 * margin
            y2 = y1 + CELL_SIZE - 2 * margin
            cell = board_image[y1:y2, x1:x2]
            cells.append(cell)

    return cells


def split_board_with_positions(board_image: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    cells = []
    margin = int(CELL_SIZE * (1 - CROP_RATIO) / 2)

    for row in range(ROWS):
        for col in range(COLS):
            x1 = col * CELL_SIZE + margin
            y1 = row * CELL_SIZE + margin
            x2 = x1 + CELL_SIZE - 2 * margin
            y2 = y1 + CELL_SIZE - 2 * margin
            cell = board_image[y1:y2, x1:x2]
            cells.append((row, col, cell))

    return cells
