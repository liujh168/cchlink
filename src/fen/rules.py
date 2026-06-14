from __future__ import annotations

from src.fen.fen_builder import EMPTY_IDX
from src.geometry import COLS

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


def legalize_predictions(
    predictions: list[int], confidences: list[float], replace_threshold: float = 0.82
) -> list[int]:
    """用基础象棋规则移除低置信度且明显不合法的预测。

    当前规则只处理可以安全判断的约束：帅将必须位于各自九宫、每类棋子数量不能
    超过理论上限。棋盘方向根据红黑棋子的平均行位置推断；高置信度预测会被保留，
    避免规则层过度修改模型结果。
    """
    result = predictions.copy()
    red_rows = [index // COLS for index, piece in enumerate(result) if 0 <= piece <= 6]
    black_rows = [index // COLS for index, piece in enumerate(result) if 7 <= piece <= 13]
    red_on_top = bool(
        red_rows
        and black_rows
        and sum(red_rows) / len(red_rows) < sum(black_rows) / len(black_rows)
    )
    red_palace_rows = range(0, 3) if red_on_top else range(7, 10)
    black_palace_rows = range(7, 10) if red_on_top else range(0, 3)

    for index, piece in enumerate(result):
        row, col = divmod(index, COLS)
        if piece == 0 and not (row in red_palace_rows and 3 <= col <= 5):
            if confidences[index] < replace_threshold:
                result[index] = EMPTY_IDX
        elif piece == 7 and not (row in black_palace_rows and 3 <= col <= 5):
            if confidences[index] < replace_threshold:
                result[index] = EMPTY_IDX

    for piece, maximum in MAX_COUNTS.items():
        indices = [index for index, value in enumerate(result) if value == piece]
        if len(indices) <= maximum:
            continue
        for index in sorted(indices, key=lambda item: confidences[item])[: len(indices) - maximum]:
            result[index] = EMPTY_IDX

    return result
