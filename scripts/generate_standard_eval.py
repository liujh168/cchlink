import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_board import generate_random_midgame, render_board  # noqa: E402
from src.standard_board import (  # noqa: E402
    STANDARD_INITIAL_LAYOUT,
    empty_layout,
    layout_to_fen,
)
from src.synthetic_scene import place_board_in_scene  # noqa: E402

FIELDS = ["path", "expected_fen", "source", "style", "layout_type", "seed", "orientation"]
SOURCE = "standard-eval-v2"
EVAL_STYLES = ("classic", "wood", "plastic")


def _layout_for_index(index: int, rng: random.Random):
    kind = index % 20
    if kind < 3:
        return [row.copy() for row in STANDARD_INITIAL_LAYOUT], "initial"
    if kind < 5:
        return empty_layout(), "empty"
    state = random.getstate()
    random.setstate(rng.getstate())
    try:
        layout = generate_random_midgame()
        rng.setstate(random.getstate())
    finally:
        random.setstate(state)
    return layout, "midgame"


def generate_eval_set(output: Path, manifest: Path, count: int = 60, seed: int = 90000) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，请使用新的目录: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    styles = EVAL_STYLES
    rows = []
    for index in range(count):
        item_seed = seed + index
        style = styles[index % len(styles)]
        layout, layout_type = _layout_for_index(index, rng)
        orientation = "red_top" if index % 2 else "red_bottom"
        board = np.asarray(render_board(layout, style=style, scale=2))
        if orientation == "red_top":
            board = np.rot90(board, 2).copy()
        scene = place_board_in_scene(board, style=style, seed=item_seed)
        filename = f"board_{index:03d}_{style}_{layout_type}.png"
        cv2.imwrite(str(output / filename), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))
        try:
            manifest_path = (output / filename).resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            manifest_path = (output / filename).resolve()
        rows.append(
            {
                "path": manifest_path.as_posix(),
                "expected_fen": layout_to_fen(layout),
                "source": SOURCE,
                "style": style,
                "layout_type": layout_type,
                "seed": item_seed,
                "orientation": orientation,
            }
        )
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"count": count, "seed": seed, "styles": list(styles), "source": SOURCE}
    (output / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="生成独立标准合成端到端评估集")
    parser.add_argument("--output", default="evaluation/generated_v2", help="评估图片目录")
    parser.add_argument("--manifest", default="evaluation/standard_manifest.csv", help="评估清单")
    parser.add_argument("--count", type=int, default=60, help="评估棋盘数量")
    parser.add_argument("--seed", type=int, default=90000, help="评估集随机种子")
    args = parser.parse_args()
    summary = generate_eval_set(Path(args.output), Path(args.manifest), args.count, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
