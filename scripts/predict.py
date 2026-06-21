import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.pipeline import Pipeline  # noqa: E402


def build_model_config(primary_model: str, ensemble_models: list[str], weights: list[float] | None):
    models = [primary_model, *ensemble_models]
    if not weights:
        return models, None
    if len(weights) != len(models):
        raise ValueError("--ensemble-weight 数量必须等于主模型 + ensemble 模型总数")
    return models, weights


def main():
    parser = argparse.ArgumentParser(description="中国象棋棋盘 FEN 识别")
    parser.add_argument("image", help="棋盘图片路径")
    parser.add_argument("--model", "-m", required=True, help="训练好的模型权重路径 (*.pth)")
    parser.add_argument(
        "--ensemble-model",
        action="append",
        default=[],
        help="额外参与概率平均的模型权重路径，可重复传入",
    )
    parser.add_argument(
        "--ensemble-weight",
        action="append",
        type=float,
        help="模型融合权重；数量需等于主模型加 ensemble 模型总数",
    )
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--backbone", default="mobilenet_v3_small", help="模型骨干网络")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印详细信息")
    parser.add_argument("--visualize-dir", help="保存原图和校正棋盘可视化结果的目录")
    parser.add_argument("--debug-dir", help="保存完整中间调试产物的目录")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"无法读取图片: {args.image}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        model_paths, model_weights = build_model_config(
            args.model, args.ensemble_model, args.ensemble_weight
        )
        pipeline = Pipeline(
            model_path=model_paths,
            device=args.device,
            backbone=args.backbone,
            model_weights=model_weights,
        )
    except ValueError as error:
        print(f"参数错误: {error}", file=sys.stderr)
        sys.exit(1)

    try:
        artifact_name = Path(args.image).stem
        visualize_dir = Path(args.visualize_dir) / artifact_name if args.visualize_dir else None
        debug_dir = Path(args.debug_dir) / artifact_name if args.debug_dir else None
        analysis = pipeline.analyze(
            image_rgb,
            debug_dir=debug_dir,
            visualize_dir=visualize_dir,
        )
        if args.verbose:
            result = analysis.to_dict()
            print("=" * 50)
            print("FEN:", result["fen"])
            print("原始 FEN:", result["raw_fen"])
            print("照片方向:", result["orientation"])
            print(f"棋盘置信度: {result['board_confidence']:.3f}")
            print(f"网格置信度: {result['grid_confidence']:.3f}")
            print("警告:", ", ".join(item["code"] for item in result["warnings"]) or "无")
            print("=" * 50)
        else:
            print(analysis.fen)
    except RuntimeError as error:
        print(f"拒识: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
