import cv2
import numpy as np


BOARD_WIDTH = 450
BOARD_HEIGHT = 500
WARP_PAD = 30


def warp_board(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst_w = BOARD_WIDTH + 2 * WARP_PAD
    dst_h = BOARD_HEIGHT + 2 * WARP_PAD

    dst_pts = np.array(
        [
            [WARP_PAD, WARP_PAD],
            [WARP_PAD + BOARD_WIDTH - 1, WARP_PAD],
            [WARP_PAD + BOARD_WIDTH - 1, WARP_PAD + BOARD_HEIGHT - 1],
            [WARP_PAD, WARP_PAD + BOARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
    return warped
