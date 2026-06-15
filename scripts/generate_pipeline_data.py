import os
import sys
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from generate_board import (
    render_board, INITIAL_LAYOUT, PIECE_CHARS,
    ROWS, COLS, CELL, BOARD_W, BOARD_H,
    apply_perspective, apply_rotation, apply_lighting,
)
from src.preprocess.board_detector import detect_board_corners
from src.preprocess.perspective import warp_board, WARP_PAD
from src.segmentation.grid_splitter import _snap_grid, _detect_grid_lines
from src.recognition.dataset import CLASS_TO_IDX

RED_PIECES = ["红帅", "红仕", "红相", "红俥", "红马", "红炮", "红兵"]
BLACK_PIECES = ["黑将", "黑士", "黑象", "黑车", "黑马", "黑炮", "黑卒"]
ALL_PIECES = RED_PIECES + BLACK_PIECES


def generate_random_midgame():
    layout = [[None] * COLS for _ in range(ROWS)]
    placed = set()
    count = random.randint(8, 25)

    for _ in range(count):
        piece = random.choice(ALL_PIECES)
        if piece in placed:
            continue
        attempts = 0
        while attempts < 100:
            row = random.randint(0, ROWS - 1)
            col = random.randint(0, COLS - 1)

            if (piece in ("红帅", "黑将") and (3 <= col <= 5) and
                    ((piece == "黑将" and 0 <= row <= 2) or (piece == "红帅" and 7 <= row <= 9))):
                if layout[row][col] is None:
                    layout[row][col] = piece
                    placed.add(piece)
                    break
            elif piece.startswith("红") and row < 5:
                attempts += 1
                continue
            elif piece.startswith("黑") and row > 4:
                attempts += 1
                continue
            elif piece not in ("红帅", "黑将"):
                if layout[row][col] is None:
                    layout[row][col] = piece
                    placed.add(piece)
                    break
            attempts += 1

    return layout


def distort_to_array(pil_img):
    img = np.array(pil_img)
    h, w = img.shape[:2]

    angle = random.uniform(-20, 20)
    bg_color = tuple(random.randint(0, 60) for _ in range(3))

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=bg_color)

    shift = random.randint(10, 50)
    src_pts = np.float32([
        [shift, shift],
        [w - 1 - shift, random.randint(0, shift)],
        [w - 1 - random.randint(0, shift), h - 1 - shift],
        [random.randint(0, shift), h - 1 - random.randint(0, shift)],
    ])
    dst_size = 600
    dst_pts = np.float32([
        [30, 30],
        [dst_size - 1 - 30, 30],
        [dst_size - 1 - 30, dst_size - 1 - 30],
        [30, dst_size - 1 - 30],
    ])

    M2 = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(rotated, M2, (dst_size, dst_size),
                                  borderValue=bg_color)

    brightness = random.uniform(0.5, 1.5)
    result = np.clip(result.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    return result


def extract_cells_from_board(board_img, layout):
    corners = detect_board_corners(board_img)
    if corners is None:
        return []

    warped = warp_board(board_img, corners)
    hw, ww = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    h_lines = _snap_grid(gray, axis='h', count=ROWS + 1, size=hw)
    v_lines = _snap_grid(gray, axis='v', count=COLS + 1, size=ww)

    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            y1 = max(0, h_lines[r])
            y2 = min(ww, h_lines[r + 1])
            x1 = max(0, v_lines[c])
            x2 = min(hw, v_lines[c + 1])
            cell = warped[y1:y2, x1:x2]
            piece = layout[r][c]
            label_name = piece if piece else "空"
            cells.append((label_name, cell))

    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/pieces_pipeline", help="输出目录")
    parser.add_argument("--num-boards", "-n", type=int, default=1000, help="生成的棋盘数量")
    args = parser.parse_args()

    output_dir = args.output
    for class_name in CLASS_TO_IDX:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    counters = {name: 0 for name in CLASS_TO_IDX}
    total_boards = 0

    for i in range(args.num_boards):
        seed = random.randint(0, 999999)
        random.seed(seed)
        np.random.seed(seed)

        layout = generate_random_midgame()
        board_img = render_board(layout)
        distorted = distort_to_array(board_img)
        distorted_bgr = distorted

        cells = extract_cells_from_board(distorted_bgr, layout)
        for label_name, cell in cells:
            if cell.size == 0:
                continue
            idx = counters[label_name]
            Image.fromarray(cell).save(
                os.path.join(output_dir, label_name, f"{idx:05d}.png")
            )
            counters[label_name] += 1

        total_boards += 1
        if (i + 1) % 50 == 0:
            print(f"  已处理: {i + 1}/{args.num_boards}")

    print(f"\n总共处理 {total_boards} 个棋盘")
    print("\n各类别统计:")
    for class_name in CLASS_TO_IDX:
        print(f"  {class_name:6s}: {counters[class_name]} 张")

    total = sum(counters.values())
    print(f"\n总计: {total} 张图片")


if __name__ == "__main__":
    main()