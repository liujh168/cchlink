import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_SITE = os.path.join(PROJECT_ROOT, ".venv", "Lib", "site-packages")
if os.path.exists(VENV_SITE) and VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

import random
import numpy as np
import cv2
from collections import defaultdict
from scripts.generate_board import (
    INITIAL_LAYOUT, render_board, COLS, ROWS, generate_random_midgame,
)
from src.pipeline import Pipeline
from src.fen.fen_builder import IDX_TO_FEN, IDX_TO_NAME
from src.recognition.dataset import CLASS_TO_IDX
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "checkpoint_v8.pth")

FEN_TO_NAME = {fen_char: IDX_TO_NAME[idx] for idx, fen_char in IDX_TO_FEN.items() if fen_char}


def layout_to_fen(layout):
    rows = []
    for row in range(ROWS):
        s = ""
        empty = 0
        for col in range(COLS):
            piece = layout[row][col]
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    s += str(empty)
                    empty = 0
                idx = CLASS_TO_IDX[piece]
                s += IDX_TO_FEN[idx]
        if empty > 0:
            s += str(empty)
        rows.append(s)
    return "/".join(rows)


def fen_to_layout(fen):
    layout = [[None] * COLS for _ in range(ROWS)]
    if not fen:
        return layout
    fen_rows = fen.split("/")
    for r, row_str in enumerate(fen_rows):
        if r >= ROWS:
            break
        c = 0
        for ch in row_str:
            if ch.isdigit():
                c += int(ch)
            else:
                if c < COLS:
                    name = FEN_TO_NAME.get(ch)
                    layout[r][c] = name
                c += 1
    return layout


def eval_layout(layout_a, layout_b):
    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in range(ROWS):
        for c in range(COLS):
            a = layout_a[r][c] or "空"
            b = layout_b[r][c] or "空"
            total += 1
            if a == b:
                correct += 1
            per_class[a]["total"] += 1
            if a == b:
                per_class[a]["correct"] += 1
    return correct, total, per_class


def distort_image(pil_img):
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

    return warped


def main():
    eval_dir = os.path.join(PROJECT_ROOT, "data", "raw", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    random.seed(123)

    all_correct = 0
    all_total = 0
    all_per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    total_exact = 0

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    pipeline = Pipeline(model_path=MODEL_PATH, device=device)

    num_tests = 20
    for i in range(num_tests):
        use_initial = random.random() < 0.3
        if use_initial:
            layout = INITIAL_LAYOUT
        else:
            layout = generate_random_midgame()

        board_pil = render_board(layout)
        distorted = distort_image(board_pil)

        img_path = os.path.join(eval_dir, f"test_{i:03d}.png")
        cv2.imwrite(img_path, distorted)

        expected_fen = layout_to_fen(layout)

        try:
            result = pipeline.run_verbose(distorted)
            fen = result["fen"]
        except Exception as e:
            print(f"[!] test_{i:03d} error: {e}")
            continue

        predicted_layout = fen_to_layout(fen)

        correct, total, per_class = eval_layout(layout, predicted_layout)
        all_correct += correct
        all_total += total
        for k, v in per_class.items():
            all_per_class[k]["correct"] += v["correct"]
            all_per_class[k]["total"] += v["total"]

        exact = fen == expected_fen
        if exact:
            total_exact += 1

        status = "OK" if exact else "NG"
        print(f"[{status}] test_{i:03d} accuracy={correct}/{total} "
              f"expected={expected_fen} got={fen}")

    print(f"\n=== 总体评估 ({num_tests} 张) ===")
    print(f"完全匹配: {total_exact}/{num_tests}")
    print(f"总准确率: {all_correct}/{all_total} = {100 * all_correct / all_total:.1f}%")

    print(f"\n各分类准确率:")
    for cls in sorted(all_per_class.keys(),
                      key=lambda x: (x != "空", x)):
        v = all_per_class[cls]
        acc = 100 * v["correct"] / v["total"] if v["total"] > 0 else 0
        print(f"  {cls}: {v['correct']}/{v['total']} ({acc:.1f}%)")


if __name__ == "__main__":
    main()