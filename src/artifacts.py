from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.analysis import AnalysisResult
from src.geometry import COLS, ROWS


def _font(size: int):
    for path in (
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_original_overlay(image: np.ndarray, result: AnalysisResult) -> np.ndarray:
    """在原图上绘制识别到的棋盘四边形及整体置信度。"""
    canvas = image.copy()
    corners = np.asarray(result.corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [corners], True, (0, 220, 0), 3, cv2.LINE_AA)
    label = f"board={result.board_confidence:.3f} grid={result.grid_confidence:.3f}"
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
    return canvas


def render_board_overlay(board: np.ndarray, result: AnalysisResult) -> np.ndarray:
    """在规范方向棋盘上叠加交点、棋子名称、置信度和警告位置。"""
    warning_positions = {position for warning in result.warnings for position in warning.positions}
    pil_image = Image.fromarray(board.copy())
    draw = ImageDraw.Draw(pil_image)
    font = _font(13)
    for cell, y, x in zip(
        result.cells,
        [y for y in result.row_positions for _ in result.col_positions],
        result.col_positions * len(result.row_positions),
    ):
        color = (230, 40, 40) if (cell.row, cell.col) in warning_positions else (20, 150, 20)
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        if cell.name != "空":
            draw.text((x + 6, y - 18), f"{cell.name} {cell.confidence:.2f}", fill=color, font=font)
    return np.asarray(pil_image)


def build_contact_sheet(cells: list[np.ndarray], size: int = 64) -> np.ndarray:
    """将 90 个交点图块按棋盘位置拼成一张接触表。"""
    sheet = np.zeros((ROWS * size, COLS * size, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        row, col = divmod(index, COLS)
        resized = cv2.resize(cell, (size, size), interpolation=cv2.INTER_AREA)
        sheet[row * size : (row + 1) * size, col * size : (col + 1) * size] = resized
    return sheet


def save_visualizations(
    output_dir: str | Path,
    image: np.ndarray,
    board: np.ndarray,
    result: AnalysisResult,
) -> None:
    """保存原图定位叠加图和规范方向棋盘结果叠加图。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(output / "original_overlay.png"),
        cv2.cvtColor(render_original_overlay(image, result), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(output / "board_overlay.png"),
        cv2.cvtColor(render_board_overlay(board, result), cv2.COLOR_RGB2BGR),
    )


def save_debug_artifacts(
    output_dir: str | Path,
    image: np.ndarray,
    board: np.ndarray,
    cells: list[np.ndarray],
    result: AnalysisResult,
) -> None:
    """保存复现一次识别过程所需的中间图像和完整结构化结果。"""
    output = Path(output_dir)
    save_visualizations(output, image, board, result)
    # 使用稳定的专用文件名保留本次选中的候选棋盘，方便诊断脚本直接读取。
    cv2.imwrite(
        str(output / "board_candidate.png"),
        cv2.cvtColor(render_original_overlay(image, result), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(output / "rectified_board.png"), cv2.cvtColor(board, cv2.COLOR_RGB2BGR))

    grid = board.copy()
    for y in result.row_positions:
        cv2.line(grid, (0, y), (grid.shape[1] - 1, y), (0, 200, 0), 1)
    for x in result.col_positions:
        cv2.line(grid, (x, 0), (x, grid.shape[0] - 1), (0, 200, 0), 1)
    cv2.imwrite(str(output / "grid_overlay.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(output / "cells_contact_sheet.png"),
        cv2.cvtColor(build_contact_sheet(cells), cv2.COLOR_RGB2BGR),
    )
    (output / "analysis.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
