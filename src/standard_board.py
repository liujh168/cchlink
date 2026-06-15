from __future__ import annotations

ROWS = 10
COLS = 9
EMPTY_NAME = "空"
STANDARD_INITIAL_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
PIECE_TO_FEN = {
    "红帅": "K",
    "红仕": "A",
    "红相": "B",
    "红俥": "R",
    "红马": "N",
    "红炮": "C",
    "红兵": "P",
    "黑将": "k",
    "黑士": "a",
    "黑象": "b",
    "黑车": "r",
    "黑马": "n",
    "黑炮": "c",
    "黑卒": "p",
}

STANDARD_INITIAL_LAYOUT = [
    ["黑车", "黑马", "黑象", "黑士", "黑将", "黑士", "黑象", "黑马", "黑车"],
    [None] * COLS,
    [None, "黑炮", None, None, None, None, None, "黑炮", None],
    ["黑卒", None, "黑卒", None, "黑卒", None, "黑卒", None, "黑卒"],
    [None] * COLS,
    [None] * COLS,
    ["红兵", None, "红兵", None, "红兵", None, "红兵", None, "红兵"],
    [None, "红炮", None, None, None, None, None, "红炮", None],
    [None] * COLS,
    ["红俥", "红马", "红相", "红仕", "红帅", "红仕", "红相", "红马", "红俥"],
]


def empty_layout() -> list[list[None]]:
    return [[None] * COLS for _ in range(ROWS)]


def clone_layout(layout: list[list[str | None]]) -> list[list[str | None]]:
    return [row.copy() for row in layout]


def rotate_layout(layout: list[list[str | None]]) -> list[list[str | None]]:
    return [list(reversed(row)) for row in reversed(layout)]


def layout_to_fen(layout: list[list[str | None]]) -> str:
    rows = []
    for row in layout:
        empty = 0
        encoded = ""
        for piece in row:
            if piece is None:
                empty += 1
                continue
            if empty:
                encoded += str(empty)
                empty = 0
            encoded += PIECE_TO_FEN[piece]
        if empty:
            encoded += str(empty)
        rows.append(encoded)
    return "/".join(rows)
