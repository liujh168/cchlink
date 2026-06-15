from __future__ import annotations

from collections import Counter

from src.analysis import AnalysisWarning, Correction, Orientation
from src.fen.fen_builder import EMPTY_IDX
from src.geometry import COLS, ROWS

MAX_COUNTS = {
    0: 1,
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    6: 5,
    7: 1,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 5,
}

RED_PALACE = {(row, col) for row in range(7, 10) for col in range(3, 6)}
BLACK_PALACE = {(row, col) for row in range(0, 3) for col in range(3, 6)}
RED_ADVISOR_POINTS = {(9, 3), (9, 5), (8, 4), (7, 3), (7, 5)}
BLACK_ADVISOR_POINTS = {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)}
RED_ELEPHANT_POINTS = {(9, 2), (9, 6), (7, 0), (7, 4), (7, 8), (5, 2), (5, 6)}
BLACK_ELEPHANT_POINTS = {(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)}


def rotate_predictions(values: list) -> list:
    """将 10x9 棋盘整体旋转 180 度。"""
    if len(values) != ROWS * COLS:
        raise ValueError(f"棋盘数据应为 {ROWS * COLS} 个元素")
    return list(reversed(values))


def _orientation_score(predictions: list[int], red_bottom: bool) -> float:
    oriented = predictions if red_bottom else rotate_predictions(predictions)
    score = 0.0
    red_rows = []
    black_rows = []
    for index, piece in enumerate(oriented):
        row, col = divmod(index, COLS)
        position = (row, col)
        if 0 <= piece <= 6:
            red_rows.append(row)
        elif 7 <= piece <= 13:
            black_rows.append(row)
        if piece == 0:
            score += 6 if position in RED_PALACE else -6
        elif piece == 7:
            score += 6 if position in BLACK_PALACE else -6
        elif piece == 1:
            score += 2 if position in RED_ADVISOR_POINTS else -1
        elif piece == 8:
            score += 2 if position in BLACK_ADVISOR_POINTS else -1
        elif piece == 2:
            score += 2 if position in RED_ELEPHANT_POINTS else -1
        elif piece == 9:
            score += 2 if position in BLACK_ELEPHANT_POINTS else -1
    if red_rows and black_rows:
        score += 3 if sum(red_rows) / len(red_rows) > sum(black_rows) / len(black_rows) else -3
    return score


def detect_orientation(predictions: list[int], minimum_margin: float = 2.0) -> Orientation:
    """结合帅将、仕相合法位置与双方平均行位置判断照片方向。"""
    bottom_score = _orientation_score(predictions, red_bottom=True)
    top_score = _orientation_score(predictions, red_bottom=False)
    if abs(bottom_score - top_score) < minimum_margin:
        return "unknown"
    return "red_bottom" if bottom_score > top_score else "red_top"


def normalize_orientation(
    predictions: list[int], confidences: list[float], orientation: Orientation
) -> tuple[list[int], list[float]]:
    """将预测统一规范为黑方在上、红方在下。"""
    if orientation == "red_top":
        return rotate_predictions(predictions), rotate_predictions(confidences)
    return predictions.copy(), confidences.copy()


def legalize_predictions_with_corrections(
    predictions: list[int], confidences: list[float], replace_threshold: float = 0.82
) -> tuple[list[int], list[Correction]]:
    """仅修正低置信度且违反基础规则的预测，并记录每次修改。"""
    result = predictions.copy()
    corrections = []

    for index, piece in enumerate(result):
        row, col = divmod(index, COLS)
        illegal_king = piece == 0 and (row, col) not in RED_PALACE
        illegal_general = piece == 7 and (row, col) not in BLACK_PALACE
        if (illegal_king or illegal_general) and confidences[index] < replace_threshold:
            corrections.append(Correction(row, col, piece, EMPTY_IDX, "帅将位于九宫之外"))
            result[index] = EMPTY_IDX

    for piece, maximum in MAX_COUNTS.items():
        indices = [index for index, value in enumerate(result) if value == piece]
        if len(indices) <= maximum:
            continue
        removable = sorted(indices, key=lambda item: confidences[item])
        for index in removable[: len(indices) - maximum]:
            if confidences[index] >= replace_threshold:
                continue
            row, col = divmod(index, COLS)
            corrections.append(Correction(row, col, piece, EMPTY_IDX, "棋子数量超过理论上限"))
            result[index] = EMPTY_IDX

    return result, corrections


def legalize_predictions(
    predictions: list[int], confidences: list[float], replace_threshold: float = 0.82
) -> list[int]:
    """兼容旧接口，仅返回规则修正后的预测列表。"""
    result, _ = legalize_predictions_with_corrections(predictions, confidences, replace_threshold)
    return result


def _positions(predictions: list[int], piece: int) -> list[tuple[int, int]]:
    return [divmod(index, COLS) for index, value in enumerate(predictions) if value == piece]


def _warning(code: str, severity: str, message: str, positions=None) -> AnalysisWarning:
    return AnalysisWarning(code, severity, message, positions or [])


def _pawn_is_illegal(piece: int, row: int, col: int) -> bool:
    if piece == 6:
        return row >= 7 or (row in (5, 6) and col % 2 == 1)
    return row <= 2 or (row in (3, 4) and col % 2 == 1)


def validate_position(predictions: list[int], orientation: Orientation) -> list[AnalysisWarning]:
    """校验规范方向下单帧可以确定的静态象棋规则。"""
    warnings = []
    if orientation == "unknown":
        warnings.append(_warning("orientation_unknown", "warning", "棋盘方向证据不足或相互冲突"))

    counts = Counter(piece for piece in predictions if piece != EMPTY_IDX)
    for piece, expected_name in ((0, "红帅"), (7, "黑将")):
        positions = _positions(predictions, piece)
        if len(positions) != 1:
            warnings.append(
                _warning(
                    "king_count" if piece == 0 else "general_count",
                    "error",
                    f"{expected_name}数量应为 1，当前为 {len(positions)}",
                    positions,
                )
            )

    for piece, maximum in MAX_COUNTS.items():
        if counts[piece] > maximum:
            warnings.append(
                _warning(
                    "piece_count_exceeded",
                    "error",
                    f"类别 {piece} 数量 {counts[piece]} 超过上限 {maximum}",
                    _positions(predictions, piece),
                )
            )

    legal_sets = {
        0: RED_PALACE,
        7: BLACK_PALACE,
        1: RED_ADVISOR_POINTS,
        8: BLACK_ADVISOR_POINTS,
        2: RED_ELEPHANT_POINTS,
        9: BLACK_ELEPHANT_POINTS,
    }
    codes = {
        0: "king_outside_palace",
        7: "general_outside_palace",
        1: "red_advisor_illegal",
        8: "black_advisor_illegal",
        2: "red_elephant_illegal",
        9: "black_elephant_illegal",
    }
    for piece, legal in legal_sets.items():
        illegal = [position for position in _positions(predictions, piece) if position not in legal]
        if illegal:
            warnings.append(
                _warning(codes[piece], "error", "棋子位于静态规则不允许的位置", illegal)
            )

    for piece, code in ((6, "red_pawn_illegal"), (13, "black_pawn_illegal")):
        illegal = [
            (row, col)
            for row, col in _positions(predictions, piece)
            if _pawn_is_illegal(piece, row, col)
        ]
        if illegal:
            warnings.append(_warning(code, "warning", "兵卒位于无法从初始位置到达的交点", illegal))

    red_king = _positions(predictions, 0)
    black_king = _positions(predictions, 7)
    if len(red_king) == len(black_king) == 1 and red_king[0][1] == black_king[0][1]:
        col = red_king[0][1]
        low, high = sorted((red_king[0][0], black_king[0][0]))
        blockers = [predictions[row * COLS + col] for row in range(low + 1, high)]
        if all(piece == EMPTY_IDX for piece in blockers):
            warnings.append(
                _warning(
                    "kings_facing",
                    "error",
                    "帅将之间没有棋子阻挡，形成照面",
                    red_king + black_king,
                )
            )
    return warnings
