"""生成与eval一致风格的训练数据，使用已知棋盘布局提取带标签的棋子格子。"""
import os
import sys
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from scripts.generate_board import (
    render_board, INITIAL_LAYOUT, PIECE_CHARS,
    ROWS, COLS, CELL, BOARD_W, BOARD_H,
)
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


def distort_eval_style(pil_img):
    """与eval_pipeline.py完全一致的distort_image"""
    img = np.array(pil_img)
    h, w = img.shape[:2]

    angle = random.uniform(-20, 20)
    brightness = random.uniform(0.5, 1.5)
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
        [dst_size - 31, 30],
        [dst_size - 31, dst_size - 31],
        [30, dst_size - 31],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(rotated, M, (dst_size, dst_size), borderValue=bg_color)

    warped = np.clip(warped.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        sigma = random.uniform(0.3, 1.5)
        warped = cv2.GaussianBlur(warped, (0, 0), sigma)

    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 12), warped.shape).astype(np.float32)
        warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return warped


def augment_cell(cell_img):
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        cell_img = ImageEnhance.Brightness(cell_img).enhance(factor)
    if random.random() < 0.3:
        factor = random.uniform(0.8, 1.2)
        cell_img = ImageEnhance.Contrast(cell_img).enhance(factor)
    if random.random() < 0.2:
        cell_img = cell_img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    if random.random() < 0.2:
        arr = np.array(cell_img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        cell_img = Image.fromarray(arr)
    return cell_img


def extract_cells_from_distorted(distorted, layout, cell_size=None):
    if cell_size is None:
        cell_size = CELL
    cells = []
    h, w = distorted.shape[:2]
    for r in range(ROWS):
        for c in range(COLS):
            piece = layout[r][c]
            label_name = piece if piece else "空"

            cx = int(c * (w / COLS) + (w / COLS) / 2)
            cy = int(r * (h / ROWS) + (h / ROWS) / 2)

            cell_w = int(w / COLS)
            cell_h = int(h / ROWS)
            jitter_x = random.randint(-int(cell_w * 0.2), int(cell_w * 0.2))
            jitter_y = random.randint(-int(cell_h * 0.2), int(cell_h * 0.2))

            x1 = max(0, cx - cell_w // 2 + jitter_x)
            y1 = max(0, cy - cell_h // 2 + jitter_y)
            x2 = min(w, x1 + cell_w)
            y2 = min(h, y1 + cell_h)
            x1 = max(0, x2 - cell_w)
            y1 = max(0, y2 - cell_h)

            cell = distorted[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            cells.append((label_name, cell))
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/pieces_boards_v4", help="输出目录")
    parser.add_argument("--num-boards", "-n", type=int, default=1500, help="棋盘数量")
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

        if random.random() < 0.3:
            layout = INITIAL_LAYOUT
        else:
            layout = generate_random_midgame()

        board_img = render_board(layout)
        distorted = distort_eval_style(board_img)

        cells = extract_cells_from_distorted(distorted, layout)
        for label_name, cell in cells:
            if cell.size == 0:
                continue
            cell_pil = Image.fromarray(cell)
            cell_pil = augment_cell(cell_pil)
            idx = counters[label_name]
            cell_pil.save(os.path.join(output_dir, label_name, f"{idx:05d}.png"))
            counters[label_name] += 1

        total_boards += 1
        if (i + 1) % 100 == 0:
            print(f"  已处理: {i + 1}/{args.num_boards}")

    print(f"\n总共处理 {total_boards} 个棋盘")
    print("\n各类别统计:")
    for class_name in CLASS_TO_IDX:
        print(f"  {class_name:6s}: {counters[class_name]} 张")
    print(f"\n总计: {sum(counters.values())} 张图片")


if __name__ == "__main__":
    main()