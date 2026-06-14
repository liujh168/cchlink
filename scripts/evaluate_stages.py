import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline
from src.preprocess.board_detector import detect_board
from src.preprocess.perspective import WARP_PAD, warp_board
from src.segmentation.grid_splitter import detect_grid


def fen_cells(fen):
    """将 FEN 展开为固定长度的 90 个交点，便于逐格统计准确率。"""
    cells = []
    for row in fen.split("/"):
        for char in row:
            cells.extend([""] * int(char) if char.isdigit() else [char])
    return cells


def main():
    parser = argparse.ArgumentParser(description="对固定照片回归集执行分阶段评估")
    parser.add_argument("--model", required=True, help="待评估的模型权重路径")
    parser.add_argument(
        "--manifest",
        default="evaluation/real_manifest.csv",
        help="固定照片回归集清单",
    )
    parser.add_argument("--backbone", default="mobilenet_v3_small", help="模型骨干网络")
    args = parser.parse_args()

    pipeline = Pipeline(args.model, backbone=args.backbone)
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    detected = grid_accepted = exact = correct_cells = 0
    latencies = []

    for item in rows:
        image = cv2.imread(str(PROJECT_ROOT / item["path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        board_detection = detect_board(image, min_confidence=0.22)
        if board_detection is not None:
            detected += 1
            warped = warp_board(image, board_detection.corners)
            board = warped[WARP_PAD:-WARP_PAD, WARP_PAD:-WARP_PAD]
            if detect_grid(board).confidence >= pipeline.min_grid_confidence:
                grid_accepted += 1
        try:
            result = pipeline.run_verbose(image)
            expected = fen_cells(item["expected_fen"])
            actual = fen_cells(result["fen"])
            correct_cells += sum(a == b for a, b in zip(expected, actual))
            exact += result["fen"] == item["expected_fen"]
            status = "OK" if result["fen"] == item["expected_fen"] else "NG"
        except RuntimeError as error:
            status = f"REJECTED: {error}"
        latencies.append(time.perf_counter() - started)
        print(f"{item['path']}: {status}")

    count = len(rows)
    print(f"detection_acceptance={detected}/{count}")
    print(f"grid_acceptance={grid_accepted}/{count}")
    print(f"cell_accuracy={correct_cells}/{count * 90}")
    print(f"exact_boards={exact}/{count}")
    print(f"mean_latency_seconds={sum(latencies) / max(count, 1):.4f}")


if __name__ == "__main__":
    main()
