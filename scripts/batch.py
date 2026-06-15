import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CSV_FIELDS = [
    "path",
    "status",
    "fen",
    "raw_fen",
    "orientation",
    "board_confidence",
    "grid_confidence",
    "warning_count",
    "warning_codes",
    "error",
    "elapsed_seconds",
]


def iter_images(input_dir: Path):
    """递归列出支持的图片文件，并保持稳定排序。"""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def process_image(
    path: Path,
    input_dir: Path,
    pipeline: Pipeline,
    visualize_dir: Path | None = None,
    debug_dir: Path | None = None,
) -> dict:
    """处理单张图片；任何读取或识别失败都会转换为 CSV 错误记录。"""
    relative = path.relative_to(input_dir)
    artifact_relative = relative.with_suffix("")
    started = time.perf_counter()
    record = {field: "" for field in CSV_FIELDS}
    record["path"] = relative.as_posix()
    try:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError("无法读取图片")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = pipeline.analyze(
            image,
            visualize_dir=visualize_dir / artifact_relative if visualize_dir else None,
            debug_dir=debug_dir / artifact_relative if debug_dir else None,
        )
        record.update(
            {
                "status": "ok",
                "fen": result.fen,
                "raw_fen": result.raw_fen,
                "orientation": result.orientation,
                "board_confidence": f"{result.board_confidence:.6f}",
                "grid_confidence": f"{result.grid_confidence:.6f}",
                "warning_count": len(result.warnings),
                "warning_codes": "|".join(warning.code for warning in result.warnings),
            }
        )
    except Exception as error:
        record.update({"status": "error", "error": str(error)})
    record["elapsed_seconds"] = f"{time.perf_counter() - started:.6f}"
    return record


def run_batch(
    input_dir: str | Path,
    output_csv: str | Path,
    pipeline: Pipeline,
    visualize_dir: str | Path | None = None,
    debug_dir: str | Path | None = None,
) -> list[dict]:
    """递归处理目录中的图片，失败时继续，并写出完整 CSV 汇总。"""
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise ValueError(f"输入目录不存在: {input_path}")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visual_path = Path(visualize_dir) if visualize_dir else None
    debug_path = Path(debug_dir) if debug_dir else None
    rows = [
        process_image(path, input_path, pipeline, visual_path, debug_path)
        for path in iter_images(input_path)
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="递归批量识别目录中的中国象棋棋盘图片")
    parser.add_argument("input_dir", help="待扫描的图片目录")
    parser.add_argument("--model", required=True, help="模型权重路径")
    parser.add_argument("--output", required=True, help="CSV 汇总输出路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--backbone", default="mobilenet_v3_small", help="模型骨干网络")
    parser.add_argument("--visualize-dir", help="保存每张图片可视化结果的根目录")
    parser.add_argument("--debug-dir", help="保存每张图片完整调试产物的根目录")
    args = parser.parse_args()

    pipeline = Pipeline(args.model, device=args.device, backbone=args.backbone)
    rows = run_batch(
        args.input_dir,
        args.output,
        pipeline,
        visualize_dir=args.visualize_dir,
        debug_dir=args.debug_dir,
    )
    successes = sum(row["status"] == "ok" for row in rows)
    print(f"处理完成：成功 {successes} 张，失败 {len(rows) - successes} 张")


if __name__ == "__main__":
    main()
