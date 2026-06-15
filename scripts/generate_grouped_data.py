import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_board import BOARD_STYLES, generate_random_midgame, render_board  # noqa: E402
from src.geometry import extract_intersection_patches, ideal_grid_positions  # noqa: E402
from src.standard_board import (  # noqa: E402
    STANDARD_INITIAL_LAYOUT,
    empty_layout,
    layout_to_fen,
    rotate_layout,
)

MANIFEST_FIELDS = [
    "path",
    "label",
    "group",
    "style",
    "layout_id",
    "layout_type",
    "seed",
    "orientation",
    "source",
    "fen",
]
GENERATOR_VERSION = "standard-v2"
STYLES = tuple(BOARD_STYLES)


def choose_layout(board_index: int, rng: random.Random):
    fraction = board_index % 20
    if fraction < 3:
        return [row.copy() for row in STANDARD_INITIAL_LAYOUT], "initial"
    if fraction < 5:
        return empty_layout(), "empty"

    state = random.getstate()
    random.setstate(rng.getstate())
    try:
        layout = generate_random_midgame()
        rng.setstate(random.getstate())
    finally:
        random.setstate(state)
    return layout, "midgame"


def apply_rectified_variation(board: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    brightness = rng.uniform(0.78, 1.22)
    contrast = rng.uniform(0.88, 1.12)
    varied = (board.astype(np.float32) - 127.5) * contrast + 127.5
    varied *= brightness
    varied += rng.normal(0, rng.uniform(0.5, 4.0), board.shape)
    if rng.random() < 0.25:
        varied = cv2.GaussianBlur(varied, (0, 0), rng.uniform(0.2, 0.7))
    return np.clip(varied, 0, 255).astype(np.uint8)


def generate_dataset(
    output: Path,
    num_boards: int,
    seed: int,
    empty_keep_prob: float,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，请使用新的目录: {output}")
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    records = []
    layouts = []
    for board_index in range(num_boards):
        board_seed = seed + board_index
        layout, layout_type = choose_layout(board_index, rng)
        orientation = "red_top" if rng.random() < 0.5 else "red_bottom"
        style = STYLES[board_index % len(STYLES)]
        oriented_layout = rotate_layout(layout) if orientation == "red_top" else layout
        board = np.asarray(render_board(oriented_layout, style=style))
        board = apply_rectified_variation(board, board_seed)
        rows, cols = ideal_grid_positions(board.shape[1], board.shape[0])
        patches = extract_intersection_patches(board, rows, cols)
        labels = [piece or "空" for row in oriented_layout for piece in row]
        group = f"board_{board_index:06d}"
        layout_id = f"{layout_type}_{board_index:06d}"
        fen = layout_to_fen(layout)
        layouts.append(
            {
                "group": group,
                "layout_id": layout_id,
                "layout_type": layout_type,
                "style": style,
                "orientation": orientation,
                "seed": board_seed,
                "fen": fen,
            }
        )
        for index, (patch, label) in enumerate(zip(patches, labels)):
            if label == "空" and rng.random() > empty_keep_prob:
                continue
            row, col = divmod(index, 9)
            filename = f"{group}_r{row}_c{col}.png"
            cv2.imwrite(str(images_dir / filename), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            records.append(
                {
                    "path": f"images/{filename}",
                    "label": label,
                    "group": group,
                    "style": style,
                    "layout_id": layout_id,
                    "layout_type": layout_type,
                    "seed": board_seed,
                    "orientation": orientation,
                    "source": GENERATOR_VERSION,
                    "fen": fen,
                }
            )

    with open(output / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    (output / "layouts.json").write_text(
        json.dumps(layouts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "num_boards": num_boards,
        "seed": seed,
        "empty_keep_prob": empty_keep_prob,
        "styles": list(STYLES),
        "samples": len(records),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description="生成标准棋盘按完整棋盘分组的训练图块")
    parser.add_argument("-o", "--output", default="data/pieces_grouped_v2", help="输出目录")
    parser.add_argument("-n", "--num-boards", type=int, default=2000, help="生成棋盘数量")
    parser.add_argument("--seed", type=int, default=42000, help="训练数据随机种子")
    parser.add_argument("--empty-keep-prob", type=float, default=0.18, help="空交点保留概率")
    args = parser.parse_args()

    metadata = generate_dataset(
        Path(args.output),
        num_boards=args.num_boards,
        seed=args.seed,
        empty_keep_prob=args.empty_keep_prob,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
