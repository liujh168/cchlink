import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.pipeline import Pipeline

image_path = str(PROJECT_ROOT / "data" / "raw" / "initial_00.jpg")
model_path = str(PROJECT_ROOT / "data" / "models" / "checkpoint.pth")

print(f"图片: {image_path}")
print(f"模型: {model_path}")

image = cv2.imread(image_path)
print(f"图片尺寸: {image.shape}")

pipeline = Pipeline(model_path=model_path, device="cpu")
print("Pipeline 初始化完成")

try:
    result = pipeline.run_verbose(image)
    print("=" * 60)
    print(f"FEN: {result['fen']}")
    print("=" * 60)
    print("棋盘布局:")
    for row_cells in result["grid"]:
        print(" | ".join(f"{c:4}" for c in row_cells))

    expected_fen = "rnbakabnr/1c5c1/p1p1p1p1p/9/9/9/9/P1P1P1P1P/1C5C1/RNBAKABNR"
    print("\n预期 FEN:", expected_fen)
    print("匹配:", result["fen"] == expected_fen)
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
