import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_board import INITIAL_LAYOUT, generate_random_midgame, render_board
from src.geometry import ideal_grid_positions, extract_intersection_patches


def main():
    parser = argparse.ArgumentParser(description="生成按完整棋盘分组的交点训练图块")
    parser.add_argument("-o", "--output", default="data/pieces_grouped", help="输出目录")
    parser.add_argument("-n", "--num-boards", type=int, default=500, help="生成棋盘数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--empty-keep-prob",
        type=float,
        default=0.1,
        help="空交点保留概率，用于控制空类别占比",
    )
    args = parser.parse_args()

    output = Path(args.output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    records = []

    for board_index in range(args.num_boards):
        layout = INITIAL_LAYOUT if board_index == 0 else generate_random_midgame()
        board = np.asarray(render_board(layout)).astype(np.int16)
        noise = np.random.default_rng(args.seed + board_index).normal(0, 2.0, board.shape)
        board = np.clip(board + noise, 0, 255).astype(np.uint8)
        rows, cols = ideal_grid_positions(board.shape[1], board.shape[0])
        patches = extract_intersection_patches(board, rows, cols)
        group = f"board_{board_index:06d}"
        for index, patch in enumerate(patches):
            row, col = divmod(index, 9)
            label = layout[row][col] or "空"
            if label == "空" and random.random() > args.empty_keep_prob:
                continue
            filename = f"{group}_r{row}_c{col}.png"
            cv2.imwrite(str(images_dir / filename), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            records.append({"path": f"images/{filename}", "label": label, "group": group})

    with open(output / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group"])
        writer.writeheader()
        writer.writerows(records)
    print(f"已从 {args.num_boards} 个棋盘生成 {len(records)} 个分组图块")


if __name__ == "__main__":
    main()
