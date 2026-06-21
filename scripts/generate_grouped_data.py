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
from src.geometry import (  # noqa: E402
    PATCH_SIZE,
    extract_intersection_patches,
    ideal_grid_positions,
)
from src.preprocess.perspective import WARP_PAD, warp_board  # noqa: E402
from src.standard_board import (  # noqa: E402
    STANDARD_INITIAL_LAYOUT,
    empty_layout,
    layout_to_fen,
    rotate_layout,
)
from src.synthetic_scene import place_board_in_scene  # noqa: E402

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
    "scene_augmented",
    "patch_scale",
    "patch_shift_y",
    "patch_shift_x",
    "edge_augmented",
]
GENERATOR_VERSION = "standard-v5"
STYLES = tuple(BOARD_STYLES)
EDGE_ROWS = {0, 9}
EDGE_COLS = {0, 8}


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


def apply_real_photo_variation(board: np.ndarray, seed: int) -> np.ndarray:
    """模拟真实照片/截图中常见的轻微色偏、压缩、虚焦和不均匀光照。"""
    rng = np.random.default_rng(seed)
    varied = board.astype(np.float32)

    channel_gain = rng.uniform(0.88, 1.12, size=(1, 1, 3))
    varied *= channel_gain

    height, width = varied.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x = rng.uniform(0.15 * width, 0.85 * width)
    center_y = rng.uniform(0.15 * height, 0.85 * height)
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    distance /= max(np.sqrt(width**2 + height**2), 1.0)
    vignette = 1.0 - rng.uniform(0.08, 0.22) * distance
    varied *= vignette[:, :, None]

    if rng.random() < 0.55:
        sigma = float(rng.uniform(0.25, 1.1))
        varied = cv2.GaussianBlur(varied, (0, 0), sigma)
    if rng.random() < 0.35:
        small_w = max(32, int(width * rng.uniform(0.70, 0.92)))
        small_h = max(32, int(height * rng.uniform(0.70, 0.92)))
        varied = cv2.resize(varied, (small_w, small_h), interpolation=cv2.INTER_AREA)
        varied = cv2.resize(varied, (width, height), interpolation=cv2.INTER_LINEAR)

    varied += rng.normal(0, rng.uniform(1.0, 6.0), board.shape)
    varied = np.clip(varied, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        quality = int(rng.integers(58, 90))
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(varied, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if ok:
            varied = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return varied


def rectify_scene_board(board: np.ndarray, style: str, seed: int) -> np.ndarray:
    scene, corners = place_board_in_scene(board, style=style, seed=seed, return_corners=True)
    warped = warp_board(scene, corners)
    return warped[WARP_PAD:-WARP_PAD, WARP_PAD:-WARP_PAD]


def _is_edge_cell(row: int, col: int) -> bool:
    return row in EDGE_ROWS or col in EDGE_COLS


def _choose_patch_transform(
    row: int,
    col: int,
    label: str,
    rng: random.Random,
    edge_aug_prob: float,
) -> tuple[float, int, int, bool]:
    edge_cell = _is_edge_cell(row, col)
    occupied = label != "空"
    edge_augmented = edge_cell and occupied and rng.random() < edge_aug_prob
    if edge_augmented:
        # Smaller crops make the rendered piece occupy more of the model input, which mimics
        # large real pieces. Outward shifts reproduce edge pieces clipped by imperfect framing.
        scale = rng.choice((0.68, 0.74, 0.82, 0.90, 1.00))
        outward_y = -1 if row == 0 else 1 if row == 9 else 0
        outward_x = -1 if col == 0 else 1 if col == 8 else 0
        shift_y = outward_y * rng.randint(2, 7) + rng.randint(-2, 2)
        shift_x = outward_x * rng.randint(2, 7) + rng.randint(-2, 2)
        return scale, shift_y, shift_x, True

    if occupied:
        scale = rng.choice((0.76, 0.82, 0.90, 0.98))
        jitter = 3 if edge_cell else 2
    else:
        scale = rng.choice((0.82, 0.90, 0.98, 1.05))
        jitter = 2
    return scale, rng.randint(-jitter, jitter), rng.randint(-jitter, jitter), False


def _extract_training_patch(
    image: np.ndarray,
    row_positions: list[int],
    col_positions: list[int],
    row: int,
    col: int,
    scale: float,
    shift_y: int,
    shift_x: int,
    output_size: int = PATCH_SIZE,
) -> np.ndarray:
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
    center_y = row_positions[row] + half_h + shift_y
    center_x = col_positions[col] + half_w + shift_x
    patch = padded[
        center_y - half_h : center_y + half_h + 1,
        center_x - half_w : center_x + half_w + 1,
    ]
    return cv2.resize(patch, (output_size, output_size), interpolation=cv2.INTER_AREA)


def generate_dataset(
    output: Path,
    num_boards: int,
    seed: int,
    empty_keep_prob: float,
    scene_prob: float = 0.55,
    real_photo_prob: float = 0.65,
    edge_aug_prob: float = 0.85,
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
        labels_layout = rotate_layout(layout) if orientation == "red_top" else layout
        board = np.asarray(render_board(layout, style=style))
        if orientation == "red_top":
            board = np.rot90(board, 2).copy()
        scene_augmented = rng.random() < scene_prob
        if scene_augmented:
            board = rectify_scene_board(board, style=style, seed=board_seed)
        else:
            board = apply_rectified_variation(board, board_seed)
        real_photo_augmented = rng.random() < real_photo_prob
        if real_photo_augmented:
            board = apply_real_photo_variation(board, board_seed + 17_000)
        rows, cols = ideal_grid_positions(board.shape[1], board.shape[0])
        patches = extract_intersection_patches(board, rows, cols)
        labels = [piece or "空" for row in labels_layout for piece in row]
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
            patch_scale, shift_y, shift_x, edge_augmented = _choose_patch_transform(
                row, col, label, rng, edge_aug_prob
            )
            if abs(patch_scale - 0.82) > 1e-6 or shift_y or shift_x:
                patch = _extract_training_patch(
                    board, rows, cols, row, col, patch_scale, shift_y, shift_x
                )
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
                    "scene_augmented": str(scene_augmented).lower(),
                    "patch_scale": f"{patch_scale:.2f}",
                    "patch_shift_y": str(shift_y),
                    "patch_shift_x": str(shift_x),
                    "edge_augmented": str(edge_augmented).lower(),
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
        "scene_prob": scene_prob,
        "real_photo_prob": real_photo_prob,
        "edge_aug_prob": edge_aug_prob,
        "styles": list(STYLES),
        "samples": len(records),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description="生成标准棋盘按完整棋盘分组的训练图块")
    parser.add_argument("-o", "--output", default="data/pieces_grouped_v5", help="输出目录")
    parser.add_argument("-n", "--num-boards", type=int, default=3000, help="生成棋盘数量")
    parser.add_argument("--seed", type=int, default=42000, help="训练数据随机种子")
    parser.add_argument("--empty-keep-prob", type=float, default=0.18, help="空交点保留概率")
    parser.add_argument("--scene-prob", type=float, default=0.55, help="完整场景透视增强比例")
    parser.add_argument("--real-photo-prob", type=float, default=0.65, help="真实照片风格增强比例")
    parser.add_argument("--edge-aug-prob", type=float, default=0.85, help="边缘有子图块增强比例")
    args = parser.parse_args()

    metadata = generate_dataset(
        Path(args.output),
        num_boards=args.num_boards,
        seed=args.seed,
        empty_keep_prob=args.empty_keep_prob,
        scene_prob=args.scene_prob,
        real_photo_prob=args.real_photo_prob,
        edge_aug_prob=args.edge_aug_prob,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
