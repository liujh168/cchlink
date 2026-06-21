import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from generate_board import (
    CELL,
    COLS,
    INITIAL_LAYOUT,
    ROWS,
    apply_lighting,
    apply_perspective,
    apply_rotation,
    render_board,
)
from PIL import Image

from src.recognition.dataset import CLASS_TO_IDX

RED_PIECES = ["红帅", "红仕", "红相", "红俥", "红马", "红炮", "红兵"]
BLACK_PIECES = ["黑将", "黑士", "黑象", "黑车", "黑马", "黑炮", "黑卒"]
ALL_PIECES = RED_PIECES + BLACK_PIECES


def generate_random_midgame() -> list[list[str | None]]:
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

            if (
                piece in ("红帅", "黑将")
                and (3 <= col <= 5)
                and ((piece == "黑将" and 0 <= row <= 2) or (piece == "红帅" and 7 <= row <= 9))
            ):
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


def crop_cells(board_img: Image.Image) -> list[tuple[int, int, str | None, Image.Image]]:
    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            x = c * CELL
            y = r * CELL
            cell = board_img.crop((x, y, x + CELL, y + CELL))
            cells.append((r, c, None, cell))
    return cells


def crop_cells_with_jitter(
    board_img: Image.Image, max_offset: int = 5
) -> list[tuple[int, int, str | None, Image.Image]]:
    cells = []
    bw, bh = board_img.size
    for r in range(ROWS):
        for c in range(COLS):
            cx = c * CELL + CELL // 2
            cy = r * CELL + CELL // 2
            dx = random.randint(-max_offset, max_offset)
            dy = random.randint(-max_offset, max_offset)
            x = cx + dx - CELL // 2
            y = cy + dy - CELL // 2
            x = max(0, min(x, bw - CELL))
            y = max(0, min(y, bh - CELL))
            cell = board_img.crop((x, y, x + CELL, y + CELL))
            cells.append((r, c, None, cell))
    return cells


def augment_cell(cell_img: Image.Image) -> Image.Image:
    from PIL import ImageEnhance, ImageFilter

    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        cell_img = ImageEnhance.Brightness(cell_img).enhance(factor)

    if random.random() < 0.3:
        factor = random.uniform(0.8, 1.2)
        cell_img = ImageEnhance.Contrast(cell_img).enhance(factor)

    if random.random() < 0.2:
        blur_radius = random.uniform(0.3, 1.0)
        cell_img = cell_img.filter(ImageFilter.GaussianBlur(blur_radius))

    if random.random() < 0.2:
        arr = np.array(cell_img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        cell_img = Image.fromarray(arr)

    return cell_img


def main():
    parser = argparse.ArgumentParser(description="从模拟棋盘图片中裁剪格子生成训练数据")
    parser.add_argument("--output", "-o", default="data/pieces_v3", help="输出目录")
    parser.add_argument("--num-boards", "-n", type=int, default=600, help="生成的棋盘数量")
    parser.add_argument("--include-initial", action="store_true", default=True, help="包含初始布局")
    parser.add_argument(
        "--num-initial", type=int, default=100, help="初始布局重复次数（配合光照变化）"
    )
    args = parser.parse_args()

    output_dir = args.output
    for class_name in CLASS_TO_IDX:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    counters = {name: 0 for name in CLASS_TO_IDX}
    total_boards = 0

    if args.include_initial:
        print(f"从初始布局生成 ({args.num_initial} 次模拟)...")
        for i in range(args.num_initial):
            seed = random.randint(0, 999999)
            random.seed(seed)
            np.random.seed(seed)

            board_img = render_board(INITIAL_LAYOUT)

            warped = apply_perspective(board_img)
            if random.random() < 0.7:
                warped = apply_rotation(warped)
            warped = Image.fromarray(warped)
            warped = Image.fromarray(apply_lighting(np.array(warped)))

            cells = crop_cells_with_jitter(warped)
            for r, c, _, cell_img in cells:
                piece = INITIAL_LAYOUT[r][c]
                class_name = piece if piece else "空"

                cell_img = augment_cell(cell_img)
                idx = counters[class_name]
                cell_img.save(os.path.join(output_dir, class_name, f"{idx:05d}.png"))
                counters[class_name] += 1

            total_boards += 1
            if (i + 1) % 10 == 0:
                print(f"  初始布局: {i + 1}/{args.num_initial}")

    print(f"从随机中局布局生成 ({args.num_boards} 盘)...")
    for i in range(args.num_boards):
        seed = random.randint(0, 999999)
        random.seed(seed)
        np.random.seed(seed)

        layout = generate_random_midgame()
        board_img = render_board(layout)

        warped = apply_perspective(board_img)
        if random.random() < 0.7:
            warped = apply_rotation(warped)
        warped = Image.fromarray(warped)
        warped = Image.fromarray(apply_lighting(np.array(warped)))

        cells = crop_cells_with_jitter(warped)
        for r, c, _, cell_img in cells:
            piece = layout[r][c]
            class_name = piece if piece else "空"

            cell_img = augment_cell(cell_img)
            idx = counters[class_name]
            cell_img.save(os.path.join(output_dir, class_name, f"{idx:05d}.png"))
            counters[class_name] += 1

        total_boards += 1
        if (i + 1) % 50 == 0:
            print(f"  随机布局: {i + 1}/{args.num_boards}")

    print(f"\n总共处理 {total_boards} 个棋盘")
    print("\n各类别统计:")
    for class_name in CLASS_TO_IDX:
        print(f"  {class_name:6s}: {counters[class_name]} 张")

    total = sum(counters.values())
    print(f"\n总计: {total} 张图片")


if __name__ == "__main__":
    main()
