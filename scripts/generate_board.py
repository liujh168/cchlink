import os
import sys
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CELL = 50
COLS = 9
ROWS = 10
BOARD_W = COLS * CELL
BOARD_H = ROWS * CELL

BOARD_BG = (235, 205, 150)
LINE_COLOR = (40, 20, 0)
LINE_WIDTH = 2

PIECE_CHARS = {
    "红帅": "帅", "红仕": "仕", "红相": "相", "红俥": "俥", "红马": "馬", "红炮": "炮", "红兵": "兵",
    "黑将": "将", "黑士": "士", "黑象": "象", "黑车": "車", "黑马": "馬", "黑炮": "砲", "黑卒": "卒",
}

INITIAL_LAYOUT = [
    ["黑车","黑马","黑象","黑士","黑将","黑士","黑象","黑马","黑车"],
    [None, "黑炮", None, None, None, None, None, "黑炮", None],
    ["黑卒", None, "黑卒", None, "黑卒", None, "黑卒", None, "黑卒"],
    [None]*9, [None]*9, [None]*9, [None]*9,
    ["红兵", None, "红兵", None, "红兵", None, "红兵", None, "红兵"],
    [None, "红炮", None, None, None, None, None, "红炮", None],
    ["红俥","红马","红相","红仕","红帅","红仕","红相","红马","红俥"],
]


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


def _draw_board_base(draw):
    for r in range(ROWS + 1):
        y = r * CELL
        draw.line([(0, y), (BOARD_W - CELL, y)], fill=LINE_COLOR, width=LINE_WIDTH)
    draw.line([(0, ROWS * CELL - CELL), (BOARD_W - CELL, ROWS * CELL - CELL)],
              fill=LINE_COLOR, width=LINE_WIDTH)

    for c in range(COLS + 1):
        x = c * CELL
        top_y = 0
        bot_y = ROWS * CELL - CELL
        if c in (0, COLS):
            draw.line([(x, top_y), (x, bot_y)], fill=LINE_COLOR, width=LINE_WIDTH)
        else:
            draw.line([(x, top_y), (x, 4 * CELL)], fill=LINE_COLOR, width=LINE_WIDTH)
            draw.line([(x, 5 * CELL), (x, bot_y)], fill=LINE_COLOR, width=LINE_WIDTH)

    for r1, r2 in [(0, 1), (7, 8)]:
        x1, x2 = 3 * CELL, 5 * CELL
        y1, y2 = r1 * CELL, r2 * CELL
        draw.line([(x1, y1), (x2, y2)], fill=LINE_COLOR, width=LINE_WIDTH)
        draw.line([(x2, y1), (x1, y2)], fill=LINE_COLOR, width=LINE_WIDTH)

    river_cy = 4 * CELL + CELL // 2
    font_river = _load_font(22)
    if font_river:
        draw.text((1 * CELL + 14, river_cy - 18), "楚  河", fill=LINE_COLOR, font=font_river)
        draw.text((5 * CELL + 14, river_cy - 18), "汉  界", fill=LINE_COLOR, font=font_river)


def _draw_piece(draw, cx, cy, piece_name: str, font, size=40):
    r = size // 2
    if "红" in piece_name:
        text_color = (180, 30, 30)
    else:
        text_color = (30, 80, 30)

    circle_color = (210, 180, 130)
    inner_color = (245, 220, 175)

    draw.ellipse([cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2],
                 fill=circle_color, outline=(150, 120, 80), width=2)
    draw.ellipse([cx - r + 7, cy - r + 7, cx + r - 7, cy + r - 7],
                 fill=inner_color, outline=(150, 120, 80), width=1)

    char = PIECE_CHARS[piece_name]
    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = cx - tw // 2 - bbox[0]
    ty = cy - th // 2 - bbox[1]
    draw.text((tx, ty), char, fill=text_color, font=font)


def render_board(layout: list[list[str | None]], font_size=26) -> Image.Image:
    img = Image.new("RGB", (BOARD_W, BOARD_H), BOARD_BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, BOARD_W - 1, BOARD_H - 1],
                   outline=(60, 30, 0), width=2)
    _draw_board_base(draw)

    font = _load_font(font_size)
    for r in range(ROWS):
        for c in range(COLS):
            piece = layout[r][c]
            if piece:
                cx = c * CELL + CELL // 2
                cy = r * CELL + CELL // 2
                _draw_piece(draw, cx, cy, piece, font, size=40)

    return img


def apply_perspective(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img)
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    jitter = random.uniform(0.05, 0.15)
    dst_pts = np.float32([
        [w * random.uniform(0, jitter), h * random.uniform(0, jitter)],
        [w * (1 - random.uniform(0, jitter)), h * random.uniform(0, jitter)],
        [w * (1 - random.uniform(0, jitter)), h * (1 - random.uniform(0, jitter))],
        [w * random.uniform(0, jitter), h * (1 - random.uniform(0, jitter))],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
    return warped


def apply_rotation(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
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
        max_dist = np.sqrt(w ** 2 + h ** 2)
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
    pieces = ["红帅","红仕","红相","红俥","红马","红炮","红兵",
              "黑将","黑士","黑象","黑车","黑马","黑炮","黑卒"]
    layout = [[None] * COLS for _ in range(ROWS)]
    placed = set()
    count = random.randint(8, 20)

    for _ in range(count):
        piece = random.choice(pieces)
        if piece in placed:
            continue
        attempts = 0
        while attempts < 50:
            row = random.randint(0, ROWS - 1)
            col = random.randint(0, COLS - 1)
            if (piece in ("红帅", "黑将") and (3 <= col <= 5) and
                    ((piece == "黑将" and 0 <= row <= 2) or (piece == "红帅" and 7 <= row <= 9))):
                pass
            elif piece.startswith("红") and row < 5:
                attempts += 1
                continue
            elif piece.startswith("黑") and row > 4:
                attempts += 1
                continue
            if layout[row][col] is None:
                layout[row][col] = piece
                placed.add(piece)
                break
            attempts += 1
    return layout


def main():
    parser = argparse.ArgumentParser(description="生成模拟中国象棋棋盘图片")
    parser.add_argument("--output", "-o", default="data/raw", help="输出目录")
    parser.add_argument("--num", "-n", type=int, default=8, help="生成图片数量")
    parser.add_argument("--midgame", action="store_true", help="生成随机中局布局")
    args = parser.parse_args()

    output_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output,
    ))
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
