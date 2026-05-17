import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="中国象棋棋盘 FEN 识别")
    parser.add_argument("image", help="棋盘图片路径")
    parser.add_argument("--model", "-m", required=True, help="训练好的模型权重路径 (*.pth)")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印详细信息")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"无法读取图片: {args.image}")
        sys.exit(1)

    pipeline = Pipeline(model_path=args.model, device=args.device)

    if args.verbose:
        result = pipeline.run_verbose(image)
        print("=" * 50)
        print("FEN:", result["fen"])
        print("=" * 50)
        print("棋盘布局:")
        for row_cells in result["grid"]:
            print(" | ".join(f"{c:4}" for c in row_cells))
    else:
        fen = pipeline.run(image)
        print(fen)


if __name__ == "__main__":
    main()
