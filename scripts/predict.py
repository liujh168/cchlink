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
    parser.add_argument("--backbone", default="mobilenet_v3_small", help="模型骨干网络")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印详细信息")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"无法读取图片: {args.image}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pipeline = Pipeline(model_path=args.model, device=args.device, backbone=args.backbone)

    try:
        if args.verbose:
            result = pipeline.run_verbose(image_rgb)
            print("=" * 50)
            print("FEN:", result["fen"])
            print(f"棋盘置信度: {result['board_confidence']:.3f}")
            print(f"网格置信度: {result['grid_confidence']:.3f}")
            print("=" * 50)
            print("棋盘布局:")
            for row_cells in result["grid"]:
                print(" | ".join(f"{c:4}" for c in row_cells))
        else:
            print(pipeline.run(image_rgb))
    except RuntimeError as error:
        print(f"拒识: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
