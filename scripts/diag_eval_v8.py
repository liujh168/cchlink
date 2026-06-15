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
from scripts.generate_board import (
    INITIAL_LAYOUT, render_board, COLS, ROWS, generate_random_midgame,
)
from src.preprocess.board_detector import detect_board_corners, detect_grid_homography
from src.preprocess.perspective import warp_board, WARP_PAD, BOARD_WIDTH, BOARD_HEIGHT
from src.segmentation.grid_splitter import split_board_with_positions
from src.pipeline import Pipeline, _crop_board, _extract_cells_by_homography, _extract_cells_from_corners
from src.fen.fen_builder import IDX_TO_FEN, IDX_TO_NAME
from src.recognition.dataset import CLASS_TO_IDX

MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "checkpoint_v8.pth")

FEN_TO_NAME = {fen_char: IDX_TO_NAME[idx] for idx, fen_char in IDX_TO_FEN.items() if fen_char}

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

random.seed(123)

debug_dir = os.path.join(PROJECT_ROOT, "data", "raw", "debug_v8")
os.makedirs(debug_dir, exist_ok=True)

pipeline = Pipeline(model_path=MODEL_PATH, device="cuda" if __import__("torch").cuda.is_available() else "cpu")

num_tests = 5
for i in range(num_tests):
    use_initial = random.random() < 0.3
    if use_initial:
        layout = INITIAL_LAYOUT
    else:
        layout = generate_random_midgame()

    board_pil = render_board(layout)
    img = distort_image(board_pil)

    cv2.imwrite(os.path.join(debug_dir, f"test_{i:03d}_input.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    height, width = img.shape[:2]
    print(f"\ntest_{i:03d}: image size={width}x{height}")

    corners = detect_board_corners(img)
    if corners is not None:
        print(f"  corners detected: {corners.tolist()}")
        warped = warp_board(img, corners)
        cropped = _crop_board(warped)
        cv2.imwrite(os.path.join(debug_dir, f"test_{i:03d}_warped.png"), cropped)
        cells = split_board_with_positions(cropped)
        print(f"  warped size: {cropped.shape}, cells: {len(cells)}")

        correct = 0
        for r, c, cell in cells:
            expected = layout[r][c]
            if expected:
                cv2.imwrite(os.path.join(debug_dir, f"test_{i:03d}_cell_{r}_{c}.png"), cell)
        print(f"  saved cell images to debug dir")
    else:
        print(f"  corners FAILED, trying homography...")
        H = detect_grid_homography(img)
        if H is not None:
            print(f"  homography found")
            cells_4d = _extract_cells_by_homography(img, H)
            print(f"  extracted {cells_4d.shape[0]}x{cells_4d.shape[1]} cells")
        else:
            print(f"  homography also FAILED")

    try:
        result = pipeline.run_verbose(img)
        fen = result["fen"]
        print(f"  FEN: {fen}")

        grid = result["grid"]
        correct = 0
        total = 0
        for r in range(ROWS):
            for c in range(COLS):
                expected = layout[r][c] or "空"
                predicted = grid[r][c]
                total += 1
                if expected == predicted:
                    correct += 1
        print(f"  accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    except Exception as e:
        print(f"  pipeline error: {e}")

print("done")
