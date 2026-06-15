import csv

import cv2
import numpy as np

from scripts.batch import CSV_FIELDS, run_batch
from src.analysis import AnalysisResult


class _Pipeline:
    def analyze(self, image, **kwargs):
        return AnalysisResult(
            fen="9/9/9/9/9/9/9/9/9/9",
            raw_fen="9/9/9/9/9/9/9/9/9/9",
            orientation="unknown",
            board_confidence=0.8,
            grid_confidence=0.7,
            cells=[],
            warnings=[],
            corrections=[],
            corners=[],
            row_positions=[],
            col_positions=[],
        )


def test_batch_recurses_and_records_corrupt_images(tmp_path):
    input_dir = tmp_path / "input"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    cv2.imwrite(str(nested / "ok.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    (input_dir / "broken.jpg").write_bytes(b"not an image")
    output = tmp_path / "results.csv"

    rows = run_batch(input_dir, output, _Pipeline())

    assert [row["path"] for row in rows] == ["broken.jpg", "nested/ok.png"]
    assert [row["status"] for row in rows] == ["error", "ok"]
    saved = list(csv.DictReader(open(output, encoding="utf-8-sig")))
    assert set(saved[0]) == set(CSV_FIELDS)
