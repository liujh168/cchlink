import sys

sys.path.insert(0, r"i:\cchlink")

import random
from collections import defaultdict

from scripts.eval_pipeline import (
    distort_image,
    fen_to_layout,
)
from scripts.generate_board import INITIAL_LAYOUT, generate_random_midgame, render_board
from src.pipeline import Pipeline

MODEL_PATH = r"i:\cchlink\data\models\checkpoint_v3.pth"


def main():
    pipeline = Pipeline(model_path=MODEL_PATH, device="cpu")

    confusion = defaultdict(lambda: defaultdict(int))
    total_samples = 50
    random.seed(456)

    for i in range(total_samples):
        use_initial = random.random() < 0.3
        layout = INITIAL_LAYOUT if use_initial else generate_random_midgame()
        board_pil = render_board(layout)
        distorted = distort_image(board_pil)

        try:
            result = pipeline.run_verbose(distorted)
            fen = result["fen"]
        except Exception:
            continue

        predicted_layout = fen_to_layout(fen)

        for r in range(len(layout)):
            for c in range(len(layout[0])):
                true_label = layout[r][c] or "空"
                pred_label = predicted_layout[r][c] or "空"
                confusion[true_label][pred_label] += 1

    all_classes = sorted(confusion.keys(), key=lambda x: (x != "空", x))
    print("混淆矩阵 (行=真实, 列=预测)")
    print(f"{'':>6}", end="")
    for cls in all_classes:
        print(f"{cls:>6}", end="")
    print()

    for true_cls in all_classes:
        print(f"{true_cls:>6}", end="")
        for pred_cls in all_classes:
            cnt = confusion[true_cls][pred_cls]
            print(f"{cnt:>6}", end="")
        print()

    print("\n各分类 top-3 误判:")
    for true_cls in all_classes:
        total = sum(confusion[true_cls].values())
        correct = confusion[true_cls][true_cls]
        acc = 100 * correct / total if total else 0
        errors = [(pred, cnt) for pred, cnt in confusion[true_cls].items() if pred != true_cls]
        errors.sort(key=lambda x: -x[1])
        if errors:
            top3 = ", ".join(f"{p}({c})" for p, c in errors[:3])
            print(f"  {true_cls}: acc={acc:.0f}% ({correct}/{total}), top errs -> {top3}")
        else:
            print(f"  {true_cls}: acc={acc:.0f}% ({correct}/{total}), 无误判")


if __name__ == "__main__":
    main()
