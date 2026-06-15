import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_board import COLS, INITIAL_LAYOUT, ROWS, render_board  # noqa: E402


def _scene_background(width: int, height: int, style: str) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    if style == "wood":
        base = np.zeros((height, width, 3), dtype=np.float32)
        base[:] = (72, 91, 112)
        grain = 10 * np.sin(x / 34) + 5 * np.sin(x / 93)
        base += grain[:, :, None]
    else:
        base = np.zeros((height, width, 3), dtype=np.float32)
        base[:] = (205, 209, 211)
        base += ((x + y) / (width + height) * 16)[:, :, None]
    return np.clip(base, 0, 255).astype(np.uint8)


def _place_board(board_rgb: np.ndarray, style: str, tilted: bool) -> np.ndarray:
    scene_h, scene_w = 1800, 1800
    scene = _scene_background(scene_w, scene_h, style)
    board_h, board_w = board_rgb.shape[:2]
    source = np.float32([[0, 0], [board_w - 1, 0], [board_w - 1, board_h - 1], [0, board_h - 1]])

    if tilted:
        destination = np.float32([[390, 155], [1495, 300], [1355, 1660], [210, 1470]])
    else:
        destination = np.float32([[315, 165], [1485, 195], [1515, 1630], [285, 1595]])

    transform = cv2.getPerspectiveTransform(source, destination)
    shadow_points = destination.astype(np.int32) + np.array([22, 26])
    shadow = scene.copy()
    cv2.fillConvexPoly(shadow, shadow_points, (35, 35, 35))
    scene = cv2.addWeighted(scene, 0.75, shadow, 0.25, 0)

    warped = cv2.warpPerspective(board_rgb, transform, (scene_w, scene_h))
    mask = cv2.warpPerspective(
        np.full((board_h, board_w), 255, dtype=np.uint8),
        transform,
        (scene_w, scene_h),
    )
    scene[mask > 0] = warped[mask > 0]

    yy, xx = np.mgrid[0:scene_h, 0:scene_w].astype(np.float32)
    distance = np.sqrt((xx - 420) ** 2 + (yy - 250) ** 2)
    light = 1.08 - 0.20 * distance / np.sqrt(scene_w**2 + scene_h**2)
    scene = np.clip(scene.astype(np.float32) * light[:, :, None], 0, 255).astype(np.uint8)
    return scene


def generate_previews(output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        ("initial_wood_overhead.png", "wood", False),
        ("initial_plastic_tilted.png", "plastic", True),
    ]
    paths = []
    for filename, style, tilted in specs:
        board = np.asarray(render_board(INITIAL_LAYOUT, style=style, scale=3))
        preview = _place_board(board, style=style, tilted=tilted)
        path = output / filename
        cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


def generate_empty_board_preview(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    empty_layout = [[None] * COLS for _ in range(ROWS)]
    board = np.asarray(render_board(empty_layout, style="wood", scale=3))
    preview = _place_board(board, style="wood", tilted=False)
    path = output / "empty_wood_overhead.png"
    cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return path


def main():
    parser = argparse.ArgumentParser(description="生成高清真实风格棋盘预览")
    parser.add_argument("--output", default="output/board_previews", help="预览图片输出目录")
    parser.add_argument("--empty", action="store_true", help="只生成无棋子的木质俯拍预览")
    args = parser.parse_args()
    if args.empty:
        print(generate_empty_board_preview(args.output))
        return
    for path in generate_previews(args.output):
        print(path)


if __name__ == "__main__":
    main()
