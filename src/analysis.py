from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Orientation = Literal["red_bottom", "red_top", "unknown"]
Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class AnalysisWarning:
    """识别或规则校验产生的结构化警告。"""

    code: str
    severity: Severity
    message: str
    positions: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class Correction:
    """规则后处理对单个交点做出的保守修正。"""

    row: int
    col: int
    before: int
    after: int
    reason: str


@dataclass(frozen=True)
class CellResult:
    """规范方向下单个棋盘交点的识别结果。"""

    row: int
    col: int
    prediction: int
    name: str
    confidence: float
    raw_prediction: int


@dataclass(frozen=True)
class AnalysisResult:
    """完整的单张棋盘结构化分析结果。"""

    fen: str
    raw_fen: str
    orientation: Orientation
    board_confidence: float
    grid_confidence: float
    cells: list[CellResult]
    warnings: list[AnalysisWarning]
    corrections: list[Correction]
    corners: list[list[float]]
    row_positions: list[int]
    col_positions: list[int]

    def to_dict(self) -> dict:
        """转换为可直接写入 JSON 的普通字典。"""
        return asdict(self)
