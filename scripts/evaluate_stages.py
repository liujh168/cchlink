import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline  # noqa: E402
from src.preprocess.board_detector import detect_board  # noqa: E402
from src.preprocess.perspective import WARP_PAD, warp_board  # noqa: E402
from src.segmentation.grid_splitter import detect_grid  # noqa: E402


def build_model_config(primary_model: str, ensemble_models: list[str], weights: list[float] | None):
    models = [primary_model, *ensemble_models]
    if not weights:
        return models, None
    if len(weights) != len(models):
        raise ValueError("--ensemble-weight 数量必须等于主模型 + ensemble 模型总数")
    return models, weights


def fen_cells(fen):
    cells = []
    for row in fen.split("/"):
        for char in row:
            cells.extend([""] * int(char) if char.isdigit() else [char])
    return cells


def _empty_metrics():
    return {"count": 0, "detected": 0, "grid_accepted": 0, "correct_cells": 0, "exact": 0}


def evaluate_manifest(
    model: str,
    manifest: Path,
    backbone: str = "mobilenet_v3_small",
    device: str = "cpu",
    ensemble_models: list[str] | None = None,
    model_weights: list[float] | None = None,
) -> dict:
    model_paths, weights = build_model_config(model, ensemble_models or [], model_weights)
    pipeline = Pipeline(model_paths, backbone=backbone, device=device, model_weights=weights)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    overall = _empty_metrics()
    by_style = defaultdict(_empty_metrics)
    results = []
    latencies = []
    for item in rows:
        metrics = [overall, by_style[item.get("style") or "unspecified"]]
        for metric in metrics:
            metric["count"] += 1
        path = Path(item["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            results.append({"path": item["path"], "status": "READ_ERROR"})
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        board_detection = detect_board(image, min_confidence=0.22)
        if board_detection is not None:
            for metric in metrics:
                metric["detected"] += 1
            warped = warp_board(image, board_detection.corners)
            board = warped[WARP_PAD:-WARP_PAD, WARP_PAD:-WARP_PAD]
            if detect_grid(board).confidence >= pipeline.min_grid_confidence:
                for metric in metrics:
                    metric["grid_accepted"] += 1
        try:
            result = pipeline.analyze(image)
            expected = fen_cells(item["expected_fen"])
            actual = fen_cells(result.fen)
            correct = sum(a == b for a, b in zip(expected, actual))
            exact = result.fen == item["expected_fen"]
            for metric in metrics:
                metric["correct_cells"] += correct
                metric["exact"] += int(exact)
            status = "OK" if exact else "NG"
            results.append(
                {
                    "path": item["path"],
                    "status": status,
                    "correct_cells": correct,
                    "fen": result.fen,
                    "expected_fen": item["expected_fen"],
                    "style": item.get("style", ""),
                    "layout_type": item.get("layout_type", ""),
                }
            )
        except RuntimeError as error:
            results.append({"path": item["path"], "status": "REJECTED", "error": str(error)})
        latencies.append(time.perf_counter() - started)

    def finalize(metric):
        count = metric["count"]
        return {
            **metric,
            "cell_accuracy": metric["correct_cells"] / max(count * 90, 1),
            "exact_rate": metric["exact"] / max(count, 1),
        }

    return {
        "manifest": str(manifest),
        "model": model,
        "model_paths": model_paths,
        "model_weights": weights,
        "overall": finalize(overall),
        "by_style": {style: finalize(metric) for style, metric in sorted(by_style.items())},
        "mean_latency_seconds": sum(latencies) / max(len(latencies), 1),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="对固定清单执行分阶段评估")
    parser.add_argument("--model", required=True, help="待评估的模型权重路径")
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
    parser.add_argument("--manifest", default="evaluation/standard_manifest.csv", help="评估清单")
    parser.add_argument("--backbone", default="mobilenet_v3_small", help="模型骨干网络")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--json-output", help="保存完整 JSON 报告")
    args = parser.parse_args()
    report = evaluate_manifest(
        args.model,
        Path(args.manifest),
        args.backbone,
        args.device,
        ensemble_models=args.ensemble_model,
        model_weights=args.ensemble_weight,
    )
    summary = {key: value for key, value in report.items() if key != "results"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
