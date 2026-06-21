import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from src.recognition.dataset import CLASS_TO_IDX

IMG_SIZE = 40

RED_PIECES = ["红帅", "红仕", "红相", "红俥", "红马", "红炮", "红兵"]
BLACK_PIECES = ["黑将", "黑士", "黑象", "黑车", "黑马", "黑炮", "黑卒"]

PIECE_CHARS = {
    "红帅": "帅",
    "红仕": "仕",
    "红相": "相",
    "红俥": "俥",
    "红马": "马",
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

FONT_PATHS = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simfang.ttf",
]

CIRCLE_COLORS = [
    (220, 195, 145),
    (230, 205, 155),
    (210, 185, 135),
    (225, 200, 150),
    (215, 190, 140),
]

BG_COLORS = [
    (240, 220, 170),
    (235, 215, 165),
    (245, 225, 175),
    (230, 210, 160),
    (238, 218, 168),
    (242, 222, 172),
    (228, 208, 158),
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | None:
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return None


def _draw_piece_base(draw, size: int, bg_color: tuple, circle_color: tuple, margin: int):
    draw.ellipse([0, 0, size - 1, size - 1], fill=circle_color)
    inner_margin = margin
    draw.ellipse(
        [inner_margin, inner_margin, size - 1 - inner_margin, size - 1 - inner_margin],
        fill=bg_color,
    )


def render_red_piece(char: str, font: ImageFont.FreeTypeFont, size: int = IMG_SIZE) -> Image.Image:
    bg_color = random.choice(BG_COLORS)
    circle_color = random.choice(CIRCLE_COLORS)
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    _draw_piece_base(draw, size, bg_color, circle_color, margin=3)

    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) / 2 - bbox[0]
    y = (size - th) / 2 - bbox[1]
    draw.text((x, y), char, fill=(180, 30, 30), font=font)

    return img


def render_black_piece(
    char: str, font: ImageFont.FreeTypeFont, size: int = IMG_SIZE
) -> Image.Image:
    bg_color = random.choice(BG_COLORS)
    circle_color = random.choice(CIRCLE_COLORS)
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    _draw_piece_base(draw, size, bg_color, circle_color, margin=3)

    bbox = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) / 2 - bbox[0]
    y = (size - th) / 2 - bbox[1]
    draw.text((x, y), char, fill=(30, 80, 30), font=font)

    return img


def render_empty(size: int = IMG_SIZE) -> Image.Image:
    bg_color = random.choice(BG_COLORS)
    img = Image.new("RGB", (size, size), bg_color)
    return img


def augment(img: Image.Image) -> Image.Image:
    angle = random.uniform(-8, 8)
    img = img.rotate(angle, expand=False, fillcolor=img.getpixel((0, 0)))

    scale = random.uniform(0.90, 1.10)
    new_size = int(IMG_SIZE * scale)
    img = img.resize((new_size, new_size), Image.BICUBIC)

    if new_size > IMG_SIZE:
        left = (new_size - IMG_SIZE) // 2
        img = img.crop((left, left, left + IMG_SIZE, left + IMG_SIZE))
    else:
        pad = (IMG_SIZE - new_size) // 2
        bg = img.getpixel((0, 0))
        padded = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
        padded.paste(img, (pad, pad))
        img = padded

    if random.random() < 0.3:
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        img = img.transform(
            img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=img.getpixel((0, 0))
        )

    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if random.random() < 0.3:
        factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(factor)

    if random.random() < 0.2:
        blur_radius = random.uniform(0.3, 1.0)
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    if random.random() < 0.2:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def generate_all(output_dir: str, num_per_class: int):
    for class_name in CLASS_TO_IDX:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    print(f"每类样本数: {num_per_class}")
    print(f"总数: {num_per_class * len(CLASS_TO_IDX)}\n")

    font = _load_font(24)
    if font is None:
        print("警告: 未找到中文字体，将使用默认字体（效果可能不佳）")

    for class_name, label in CLASS_TO_IDX.items():
        class_dir = os.path.join(output_dir, class_name)
        print(f"生成 {class_name}...", end=" ", flush=True)

        for i in range(num_per_class):
            if class_name in RED_PIECES:
                char = PIECE_CHARS[class_name]
                img = render_red_piece(char, font)
                img = augment(img)
            elif class_name in BLACK_PIECES:
                char = PIECE_CHARS[class_name]
                img = render_black_piece(char, font)
                img = augment(img)
            else:
                img = render_empty()
                img = augment(img)

            img.save(os.path.join(class_dir, f"{i:05d}.png"))

        print("OK")

    print(f"\n数据生成完成！共 {num_per_class * len(CLASS_TO_IDX)} 张图片")
    _print_class_summary(output_dir)


def _print_class_summary(output_dir: str):
    print("\n各类别统计:")
    for class_name in CLASS_TO_IDX:
        class_dir = os.path.join(output_dir, class_name)
        count = len([f for f in os.listdir(class_dir) if f.endswith(".png")])
        print(f"  {class_name:6s}: {count} 张")


def main():
    parser = argparse.ArgumentParser(description="生成中国象棋棋子训练数据（字体渲染 + 数据增强）")
    parser.add_argument("--output", "-o", default="data/pieces", help="输出目录")
    parser.add_argument("--num", "-n", type=int, default=400, help="每类样本数")
    args = parser.parse_args()

    output_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.output,
        )
    )
    generate_all(output_dir, args.num)


if __name__ == "__main__":
    main()
