import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.standard_board import (
    COLS,
    ROWS,
    STANDARD_INITIAL_LAYOUT,
    clone_layout,
)

CELL = 50
BOARD_W = COLS * CELL
BOARD_H = ROWS * CELL

BOARD_BG = (235, 205, 150)
LINE_COLOR = (40, 20, 0)
LINE_WIDTH = 2

BOARD_STYLES = {
    "classic": {
        "background": BOARD_BG,
        "line": LINE_COLOR,
        "border": (60, 30, 0),
        "piece_outer": (210, 180, 130),
        "piece_inner": (245, 220, 175),
        "piece_outline": (150, 120, 80),
    },
    "wood": {
        "background": (211, 154, 82),
        "line": (55, 25, 10),
        "border": (72, 34, 14),
        "piece_outer": (176, 116, 57),
        "piece_inner": (229, 183, 112),
        "piece_outline": (105, 57, 24),
    },
    "plastic": {
        "background": (239, 231, 204),
        "line": (54, 60, 60),
        "border": (74, 80, 80),
        "piece_outer": (224, 220, 204),
        "piece_inner": (248, 246, 232),
        "piece_outline": (120, 120, 110),
    },
}

PIECE_CHARS = {
    "红帅": "帅",
    "红仕": "仕",
    "红相": "相",
    "红俥": "俥",
    "红马": "馬",
    "红炮": "炮",
    "红兵": "兵",
    "黑将": "将",
    "黑士": "士",
    "黑象": "象",
    "黑车": "車",
    "黑马": "馬",
    "黑炮": "砲",
    "黑卒": "卒",
}

INITIAL_LAYOUT = STANDARD_INITIAL_LAYOUT


def _load_font(size: int) -> ImageFont.FreeTypeFont | None:
    font_paths = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simkai.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simfang.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return None


def _grid_point(row: int, col: int, scale: int = 1) -> tuple[int, int]:
    cell = CELL * scale
    offset = cell // 2
    return offset + col * cell, offset + row * cell


def _draw_position_marker(draw, row: int, col: int, scale: int, color: tuple[int, int, int]):
    """在炮位和兵卒位交点周围绘制标准直角定位标记。"""
    x, y = _grid_point(row, col, scale)
    gap = 6 * scale
    arm = 5 * scale
    width = max(1, LINE_WIDTH * scale)

    sides = []
    if col > 0:
        sides.append(-1)
    if col < COLS - 1:
        sides.append(1)

    for side in sides:
        inner_x = x + side * gap
        outer_x = x + side * (gap + arm)
        draw.line([(inner_x, y - gap), (outer_x, y - gap)], fill=color, width=width)
        draw.line([(inner_x, y - gap), (inner_x, y - gap - arm)], fill=color, width=width)
        draw.line([(inner_x, y + gap), (outer_x, y + gap)], fill=color, width=width)
        draw.line([(inner_x, y + gap), (inner_x, y + gap + arm)], fill=color, width=width)


def _draw_board_base(draw, scale: int = 1, style: str = "classic"):
    palette = BOARD_STYLES[style]
    cell = CELL * scale
    offset = cell // 2
    left_x = offset
    right_x = offset + (COLS - 1) * cell
    top_y = offset
    bot_y = offset + (ROWS - 1) * cell
    line_color = palette["line"]
    line_width = max(1, LINE_WIDTH * scale)

    for r in range(ROWS):
        y = offset + r * cell
        draw.line([(left_x, y), (right_x, y)], fill=line_color, width=line_width)

    for c in range(COLS):
        x = offset + c * cell
        if c in (0, COLS - 1):
            draw.line([(x, top_y), (x, bot_y)], fill=line_color, width=line_width)
        else:
            draw.line([(x, top_y), (x, offset + 4 * cell)], fill=line_color, width=line_width)
            draw.line([(x, offset + 5 * cell), (x, bot_y)], fill=line_color, width=line_width)

    for r1, r2 in [(0, 2), (7, 9)]:
        x1, x2 = offset + 3 * cell, offset + 5 * cell
        y1, y2 = offset + r1 * cell, offset + r2 * cell
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)
        draw.line([(x2, y1), (x1, y2)], fill=line_color, width=line_width)

    for row, col in (
        (2, 1),
        (2, 7),
        (3, 0),
        (3, 2),
        (3, 4),
        (3, 6),
        (3, 8),
        (6, 0),
        (6, 2),
        (6, 4),
        (6, 6),
        (6, 8),
        (7, 1),
        (7, 7),
    ):
        _draw_position_marker(draw, row, col, scale, line_color)

    river_cy = offset + 4 * cell + cell // 2
    font_river = _load_font(22 * scale)
    if font_river:
        draw.text(
            (cell + 14 * scale, river_cy - 18 * scale),
            "楚  河",
            fill=line_color,
            font=font_river,
        )
        draw.text(
            (5 * cell + 14 * scale, river_cy - 18 * scale),
            "汉  界",
            fill=line_color,
            font=font_river,
        )


def _draw_piece(draw, cx, cy, piece_name: str, font, size=40, style: str = "classic"):
    palette = BOARD_STYLES[style]
    r = size // 2
    if "红" in piece_name:
        text_color = (180, 30, 30)
    else:
        text_color = (30, 80, 30)

    edge = max(2, size // 20)
    inset = max(7, size // 6)
    draw.ellipse(
        [cx - r + edge, cy - r + edge, cx + r - edge, cy + r - edge],
        fill=palette["piece_outer"],
        outline=palette["piece_outline"],
        width=max(2, size // 20),
    )
    draw.ellipse(
        [cx - r + inset, cy - r + inset, cx + r - inset, cy + r - inset],
        fill=palette["piece_inner"],
        outline=palette["piece_outline"],
        width=max(1, size // 40),
    )

    char = PIECE_CHARS[piece_name]
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = cx - tw // 2 - bbox[0]
    ty = cy - th // 2 - bbox[1]
    draw.text((tx, ty), char, fill=text_color, font=font)


def _add_wood_texture(img: Image.Image, scale: int) -> Image.Image:
    array = np.asarray(img).astype(np.int16)
    height, width = array.shape[:2]
    x = np.arange(width, dtype=np.float32)
    grain = 6 * np.sin(x / (13 * scale)) + 3 * np.sin(x / (37 * scale))
    array = np.clip(array + grain[np.newaxis, :, np.newaxis], 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def render_board(
    layout: list[list[str | None]],
    font_size: int = 26,
    style: str = "classic",
    scale: int = 1,
) -> Image.Image:
    if style not in BOARD_STYLES:
        raise ValueError(f"未知棋盘样式: {style}")
    if scale < 1:
        raise ValueError("scale 必须大于等于 1")

    palette = BOARD_STYLES[style]
    img = Image.new("RGB", (BOARD_W * scale, BOARD_H * scale), palette["background"])
    if style == "wood":
        img = _add_wood_texture(img, scale)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [0, 0, BOARD_W * scale - 1, BOARD_H * scale - 1],
        outline=palette["border"],
        width=4 * scale,
    )
    _draw_board_base(draw, scale=scale, style=style)

    font = _load_font(font_size * scale)
    for r in range(ROWS):
        for c in range(COLS):
            piece = layout[r][c]
            if piece:
                cx, cy = _grid_point(r, c, scale)
                _draw_piece(draw, cx, cy, piece, font, size=40 * scale, style=style)

    return img


def _random_bg_color() -> tuple:
    return tuple(random.randint(0, 80) for _ in range(3))


def apply_perspective(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img)
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    jitter = random.uniform(0.10, 0.30)
    dst_pts = np.float32(
        [
            [w * random.uniform(0, jitter), h * random.uniform(0, jitter)],
            [w * (1 - random.uniform(0, jitter)), h * random.uniform(0, jitter)],
            [w * (1 - random.uniform(0, jitter)), h * (1 - random.uniform(0, jitter))],
            [w * random.uniform(0, jitter), h * (1 - random.uniform(0, jitter))],
        ]
    )

    bg = _random_bg_color()
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    return warped


def apply_rotation(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    angle = random.uniform(-35, 35)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    bg = _random_bg_color()
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    return rotated


def apply_lighting(img: np.ndarray) -> np.ndarray:
    if random.random() < 0.6:
        factor = random.uniform(0.6, 1.4)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        h, w = img.shape[:2]
        x0 = random.randint(0, w // 2)
        y0 = random.randint(0, h // 2)
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
        max_dist = np.sqrt(w**2 + h**2)
        vignette = 1 - 0.4 * (dist / max_dist)
        vignette = np.clip(vignette, 0.4, 1.0)
        img = np.clip(img.astype(np.float32) * vignette[:, :, np.newaxis], 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(2, 10), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def generate_test_image(layout, output_path: str, seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    board_img = render_board(layout)
    warped = apply_perspective(board_img)
    warped = apply_lighting(warped)

    cv2.imwrite(output_path, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    return warped


def generate_random_midgame() -> list[list[str | None]]:
    """从标准初始局面移除并移动棋子，生成静态合法性较高的中局样本。"""
    layout = clone_layout(STANDARD_INITIAL_LAYOUT)
    movable = [(row, col) for row in range(ROWS) for col in range(COLS) if layout[row][col]]
    random.shuffle(movable)

    for row, col in movable[: random.randint(6, 20)]:
        if layout[row][col] not in ("红帅", "黑将"):
            layout[row][col] = None

    for row, col in movable[random.randint(10, 20) : random.randint(21, 31)]:
        piece = layout[row][col]
        if piece is None or piece in ("红帅", "黑将"):
            continue
        candidates = [
            (new_row, new_col)
            for new_row in range(ROWS)
            for new_col in range(COLS)
            if layout[new_row][new_col] is None
            and (new_row >= 3 if piece.startswith("红") else new_row <= 6)
        ]
        if candidates:
            new_row, new_col = random.choice(candidates)
            layout[new_row][new_col] = piece
            layout[row][col] = None
    return layout


def main():
    parser = argparse.ArgumentParser(description="生成模拟中国象棋棋盘图片")
    parser.add_argument("--output", "-o", default="data/raw", help="输出目录")
    parser.add_argument("--num", "-n", type=int, default=8, help="生成图片数量")
    parser.add_argument("--midgame", action="store_true", help="生成随机中局布局")
    args = parser.parse_args()

    output_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.output,
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    for i in range(args.num):
        seed = 100 + i
        if args.midgame:
            layout = generate_random_midgame()
            name = f"midgame_{i:02d}.jpg"
        else:
            layout = INITIAL_LAYOUT
            name = f"initial_{i:02d}.jpg"

        path = os.path.join(output_dir, name)
        generate_test_image(layout, path, seed=seed)
        print(f"生成: {name}")

    print(f"\n共生成 {args.num} 张棋盘图片到 {output_dir}")


if __name__ == "__main__":
    main()
