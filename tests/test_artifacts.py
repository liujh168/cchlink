import json

import cv2
import numpy as np

from src.analysis import AnalysisResult, AnalysisWarning, CellResult
from src.artifacts import save_debug_artifacts, save_visualizations
from src.geometry import ideal_grid_positions


def _result():
    rows, cols = ideal_grid_positions(450, 500)
    cells = [CellResult(index // 9, index % 9, 14, "空", 0.9, 14) for index in range(90)]
    return AnalysisResult(
        fen="9/9/9/9/9/9/9/9/9/9",
        raw_fen="9/9/9/9/9/9/9/9/9/9",
        orientation="unknown",
        board_confidence=0.8,
        grid_confidence=0.7,
        cells=cells,
        warnings=[AnalysisWarning("test", "warning", "测试警告", [(0, 0)])],
        corrections=[],
        corners=[[10, 10], [440, 10], [440, 490], [10, 490]],
        row_positions=rows,
        col_positions=cols,
    )


def test_visualizations_and_debug_artifacts_are_written(tmp_path):
    image = np.full((500, 450, 3), 230, dtype=np.uint8)
    cells = [np.full((64, 64, 3), 180, dtype=np.uint8) for _ in range(90)]
    result = _result()

    save_visualizations(tmp_path / "visual", image, image, result)
    save_debug_artifacts(tmp_path / "debug", image, image, cells, result)

    original = cv2.imread(str(tmp_path / "visual" / "original_overlay.png"))
    board = cv2.imread(str(tmp_path / "visual" / "board_overlay.png"))
    assert original.shape[:2] == image.shape[:2]
    assert board.shape[:2] == image.shape[:2]
    assert cv2.imread(str(tmp_path / "debug" / "board_candidate.png")).shape[:2] == image.shape[:2]
    assert cv2.imread(str(tmp_path / "debug" / "cells_contact_sheet.png")).shape[:2] == (
        640,
        576,
    )
    payload = json.loads((tmp_path / "debug" / "analysis.json").read_text(encoding="utf-8"))
    assert payload["warnings"][0]["code"] == "test"
