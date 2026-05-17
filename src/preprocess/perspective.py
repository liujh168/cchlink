import cv2
import numpy as np


BOARD_WIDTH = 450
BOARD_HEIGHT = 500


def warp_board(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst_pts = np.array(
        [
            [0, 0],
            [BOARD_WIDTH - 1, 0],
            [BOARD_WIDTH - 1, BOARD_HEIGHT - 1],
            [0, BOARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(image, M, (BOARD_WIDTH, BOARD_HEIGHT))
    return warped
