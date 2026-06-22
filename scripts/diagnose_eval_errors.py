import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fen.fen_builder import IDX_TO_FEN, IDX_TO_NAME  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

FEN_TO_NAME = {fen: IDX_TO_NAME[idx] for idx, fen in IDX_TO_FEN.items() if fen}
EMPTY_CELL = ""
EMPTY_NAME = "空"


def fen_cells(fen: str) -> list[str]:
    cells = []
    for row in fen.split("/"):
        for char in row:
            if char.isdigit():
                cells.extend([EMPTY_CELL] * int(char))
            else:
                cells.append(char)
    if len(cells) != 90:
        raise ValueError(f"FEN 应展开为 90 格，实际为 {len(cells)}: {fen}")
    return cells


def fen_name(cell: str) -> str:
    return FEN_TO_NAME.get(cell, EMPTY_NAME)


def cell_region(row: int, col: int) -> str:
    if row in {0, 9} or col in {0, 8}:
        return "edge"
    if (0 <= row <= 2 or 7 <= row <= 9) and 3 <= col <= 5:
        return "palace"
    if row in {4, 5}:
        return "river"
    return "center"


def error_kind(expected: str, actual: str) -> str:
    if expected and not actual:
        return "piece_to_empty"
    if not expected and actual:
        return "empty_to_piece"
    if not expected and not actual:
        return "none"
    if expected.isupper() != actual.isupper():
        return "color_confusion"
    if expected.lower() == actual.lower():
        return "same_piece_color_confusion"
    return "piece_confusion"


def compare_fens(expected_fen: str, actual_fen: str) -> list[dict]:
    expected = fen_cells(expected_fen)
    actual = fen_cells(actual_fen)
    errors = []
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want == got:
            continue
        row, col = divmod(index, 9)
        errors.append(
            {
                "row": row,
                "col": col,
                "region": cell_region(row, col),
                "expected": want,
                "actual": got,
                "expected_name": fen_name(want),
                "actual_name": fen_name(got),
                "kind": error_kind(want, got),
            }
        )
    return errors


def _top(counter: Counter, limit: int) -> dict:
    return {key: value for key, value in counter.most_common(limit)}


def build_model_config(primary_model: str, ensemble_models: list[str], weights: list[float] | None):
    models = [primary_model, *ensemble_models]
    if not weights:
        return models, None
    if len(weights) != len(models):
        raise ValueError("--ensemble-weight 数量必须等于主模型 + ensemble 模型总数")
    return models, weights


def diagnose_manifest_errors(
    model: str,
    manifest: Path,
    backbone: str = "mobilenet_v3_small",
    device: str = "cpu",
    top: int = 20,
    ensemble_models: list[str] | None = None,
    model_weights: list[float] | None = None,
) -> dict:
    model_paths, weights = build_model_config(model, ensemble_models or [], model_weights)
    pipeline = Pipeline(model_paths, backbone=backbone, device=device, model_weights=weights)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))

    by_style = defaultdict(Counter)
    by_layout_type = defaultdict(Counter)
    by_expected = Counter()
    by_actual = Counter()
    by_pair = Counter()
    by_region = Counter()
    by_kind = Counter()
    by_region_kind = Counter()
    by_cell = Counter()
    board_errors = []

    total_cells = 0
    error_cells = 0
    exact = 0
    rejected = 0
    read_errors = 0

    for item in rows:
        image_path = Path(item["path"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            read_errors += 1
            board_errors.append({"path": item["path"], "status": "READ_ERROR"})
            continue

        total_cells += 90
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            result = pipeline.analyze(image)
        except RuntimeError as error:
            rejected += 1
            error_cells += 90
            board_errors.append({"path": item["path"], "status": "REJECTED", "error": str(error)})
            continue

        errors = compare_fens(item["expected_fen"], result.fen)
        exact += int(not errors)
        error_cells += len(errors)
        style = item.get("style") or "unspecified"
        layout_type = item.get("layout_type") or "unspecified"
        by_style[style]["boards"] += 1
        by_style[style]["cells"] += 90
        by_style[style]["error_cells"] += len(errors)
        by_layout_type[layout_type]["boards"] += 1
        by_layout_type[layout_type]["cells"] += 90
        by_layout_type[layout_type]["error_cells"] += len(errors)

        for error in errors:
            expected_name = error["expected_name"]
            actual_name = error["actual_name"]
            by_expected[expected_name] += 1
            by_actual[actual_name] += 1
            by_pair[f"{expected_name}->{actual_name}"] += 1
            by_region[error["region"]] += 1
            by_kind[error["kind"]] += 1
            by_region_kind[f"{error['region']}:{error['kind']}"] += 1
            by_cell[f"r{error['row']}_c{error['col']}"] += 1

        if errors:
            board_errors.append(
                {
                    "path": item["path"],
                    "style": style,
                    "layout_type": layout_type,
                    "orientation": item.get("orientation", ""),
                    "seed": item.get("seed", ""),
                    "expected_fen": item["expected_fen"],
                    "actual_fen": result.fen,
                    "error_count": len(errors),
                    "errors": errors,
                }
            )

    total_boards = len(rows)
    evaluated_boards = total_boards - read_errors - rejected
    return {
        "manifest": str(manifest),
        "model": model,
        "model_paths": model_paths,
        "model_weights": weights,
        "total_boards": total_boards,
        "evaluated_boards": evaluated_boards,
        "read_errors": read_errors,
        "rejected": rejected,
        "exact": exact,
        "exact_rate": exact / max(evaluated_boards, 1),
        "total_cells": total_cells,
        "error_cells": error_cells,
        "cell_accuracy": 1 - error_cells / max(total_cells, 1),
        "by_style": _finalize_groups(by_style),
        "by_layout_type": _finalize_groups(by_layout_type),
        "top_expected": _top(by_expected, top),
        "top_actual": _top(by_actual, top),
        "top_pairs": _top(by_pair, top),
        "top_kinds": _top(by_kind, top),
        "top_regions": _top(by_region, top),
        "top_region_kinds": _top(by_region_kind, top),
        "top_cells": _top(by_cell, top),
        "board_errors": board_errors,
    }


def _finalize_groups(groups: defaultdict[Counter]) -> dict:
    finalized = {}
    for key, value in sorted(groups.items()):
        cells = value.get("cells", value.get("boards", 0) * 90)
        error_cells = value.get("error_cells", 0)
        finalized[key] = {
            **dict(value),
            "cell_accuracy": 1 - error_cells / max(cells, 1),
        }
    return finalized


def main():
    parser = argparse.ArgumentParser(description="逐格诊断固定评估清单中的 FEN 识别错误")
    parser.add_argument("--model", required=True, help="待诊断模型权重")
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
    parser.add_argument("--top", type=int, default=20, help="聚合报告保留的 Top N")
    parser.add_argument("--json-output", help="保存完整 JSON 诊断报告")
    args = parser.parse_args()

    report = diagnose_manifest_errors(
        args.model,
        Path(args.manifest),
        backbone=args.backbone,
        device=args.device,
        top=args.top,
        ensemble_models=args.ensemble_model,
        model_weights=args.ensemble_weight,
    )
    summary = {key: value for key, value in report.items() if key != "board_errors"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
