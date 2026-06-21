from __future__ import annotations

import random

import cv2
import numpy as np


def place_board_in_scene(
    board_rgb: np.ndarray,
    style: str,
    seed: int,
    output_size: int = 720,
    return_corners: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """将完整棋盘放入带轻微透视、光照和噪声的可复现场景。"""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if style == "wood":
        background = np.full((output_size, output_size, 3), (73, 91, 108), dtype=np.float32)
    elif style == "plastic":
        background = np.full((output_size, output_size, 3), (207, 211, 213), dtype=np.float32)
    else:
        background = np.full((output_size, output_size, 3), (48, 55, 61), dtype=np.float32)

    yy, xx = np.mgrid[0:output_size, 0:output_size].astype(np.float32)
    background += (7 * np.sin(xx / 45) + 4 * np.sin(yy / 71))[:, :, None]
    scene = np.clip(background, 0, 255).astype(np.uint8)

    board_h, board_w = board_rgb.shape[:2]
    source = np.float32([[0, 0], [board_w - 1, 0], [board_w - 1, board_h - 1], [0, board_h - 1]])
    left = rng.randint(80, 125)
    right = rng.randint(80, 125)
    top = rng.randint(40, 85)
    bottom = rng.randint(40, 85)
    jitter = 35
    destination = np.float32(
        [
            [left + rng.randint(-jitter, jitter), top + rng.randint(-jitter, jitter)],
            [
                output_size - right + rng.randint(-jitter, jitter),
                top + rng.randint(-jitter, jitter),
            ],
            [
                output_size - right + rng.randint(-jitter, jitter),
                output_size - bottom + rng.randint(-jitter, jitter),
            ],
            [
                left + rng.randint(-jitter, jitter),
                output_size - bottom + rng.randint(-jitter, jitter),
            ],
        ]
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    shadow = scene.copy()
    cv2.fillConvexPoly(shadow, destination.astype(np.int32) + np.array([8, 10]), (25, 25, 25))
    scene = cv2.addWeighted(scene, 0.78, shadow, 0.22, 0)
    warped = cv2.warpPerspective(board_rgb, transform, (output_size, output_size))
    mask = cv2.warpPerspective(
        np.full((board_h, board_w), 255, dtype=np.uint8),
        transform,
        (output_size, output_size),
    )
    scene[mask > 0] = warped[mask > 0]

    brightness = rng.uniform(0.82, 1.18)
    scene = np.clip(scene.astype(np.float32) * brightness, 0, 255)
    scene += np_rng.normal(0, rng.uniform(0.5, 3.5), scene.shape)
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    if return_corners:
        return scene, destination
    return scene
