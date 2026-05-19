import sys
sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np

from src.preprocess.board_detector import detect_board_corners
from src.preprocess.perspective import warp_board
from src.segmentation.grid_splitter import split_board_with_positions
from src.recognition.predictor import PiecePredictor

image_path = r"i:\cchlink\data\raw\initial_00.jpg"
model_path = r"i:\cchlink\data\models\checkpoint_v2.pth"

image = cv2.imread(image_path)
print(f"原图尺寸: {image.shape}")

corners = detect_board_corners(image)
print(f"检测到角点: {corners is not None}")
if corners is None:
    print("失败")
    sys.exit(1)

board = warp_board(image, corners)
print(f"校正后棋盘尺寸: {board.shape}")
cv2.imwrite(r"i:\cchlink\data\raw\debug_board.jpg", board)

cells = split_board_with_positions(board)
print(f"分割格子数: {len(cells)}")

predictor = PiecePredictor(model_path=model_path, device="cpu")

all_preds = predictor.predict_grid([c for _, _, c in cells])
from src.fen.fen_builder import build_fen, IDX_TO_NAME

for row in range(10):
    line = []
    for col in range(9):
        idx = row * 9 + col
        pred = all_preds[idx]
        line.append(IDX_TO_NAME[pred])
    print(f"row{row}: {line}")

fen = build_fen(all_preds)
expected = "rnbakabnr/1c5c1/p1p1p1p1p/9/9/9/9/P1P1P1P1P/1C5C1/RNBAKABNR"
print(f"\nFEN:      {fen}")
print(f"预期:     {expected}")
print(f"匹配: {fen == expected}")

# Save debug crops
for row in range(10):
    for col in range(9):
        _, _, cell_img = cells[row * 9 + col]
        cv2.imwrite(rf"i:\cchlink\data\raw\debug_cell_r{row}_c{col}.jpg",
                     cv2.cvtColor(cell_img, cv2.COLOR_RGB2BGR))
print("已保存调试格子到 data/raw/debug_cell_*.jpg")
