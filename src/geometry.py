from __future__ import annotations

import cv2
import numpy as np

ROWS = 10
COLS = 9
BOARD_WIDTH = 450
BOARD_HEIGHT = 500
PATCH_SIZE = 64
PATCH_SCALE = 0.82


def ideal_grid_positions(width: int, height: int) -> tuple[list[int], list[int]]:
    """计算校正棋盘中 10 行、9 列交点的理想坐标。

    标准棋盘图像会在最外侧保留半个格子的边距，因此首尾交点并不位于图像边缘。
    统一使用该坐标定义，可以保证数据生成、网格检测和推理截块采用相同的几何语义。
    """
    rows = np.linspace(height / (2 * ROWS), height - height / (2 * ROWS), ROWS)
    cols = np.linspace(width / (2 * COLS), width - width / (2 * COLS), COLS)
    rows = rows.round().astype(int).tolist()
    cols = cols.round().astype(int).tolist()
    return rows, cols


def extract_intersection_patches(
    image: np.ndarray,
    row_positions: list[int],
    col_positions: list[int],
    output_size: int = PATCH_SIZE,
    scale: float = PATCH_SCALE,
) -> list[np.ndarray]:
    """围绕每个棋盘交点截取棋子图块，并统一缩放到模型输入尺寸。

    边缘交点会超出原图范围，因此先使用镜像边界扩展图像，避免边缘棋子被截断或
    使用纯色填充产生明显的训练与推理分布差异。
    """
    if len(row_positions) != ROWS or len(col_positions) != COLS:
        raise ValueError(f"交点数量应为 {ROWS}x{COLS}")

    row_spacing = float(np.median(np.diff(row_positions)))
    col_spacing = float(np.median(np.diff(col_positions)))
    half_h = max(2, int(round(row_spacing * scale / 2)))
    half_w = max(2, int(round(col_spacing * scale / 2)))

    padded = cv2.copyMakeBorder(
        image,
        half_h,
        half_h,
        half_w,
        half_w,
        borderType=cv2.BORDER_REFLECT_101,
    )
    patches = []
    for y in row_positions:
        for x in col_positions:
            center_y = y + half_h
            center_x = x + half_w
            patch = padded[
                center_y - half_h : center_y + half_h + 1,
                center_x - half_w : center_x + half_w + 1,
            ]
            patches.append(
                cv2.resize(patch, (output_size, output_size), interpolation=cv2.INTER_AREA)
            )
    return patches
