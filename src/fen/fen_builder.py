IDX_TO_FEN = {
    0: "K", 1: "A", 2: "B", 3: "R", 4: "N", 5: "C", 6: "P",
    7: "k", 8: "a", 9: "b", 10: "r", 11: "n", 12: "c", 13: "p",
    14: "",
}

IDX_TO_NAME = {
    0: "红帅", 1: "红仕", 2: "红相", 3: "红俥", 4: "红马", 5: "红炮", 6: "红兵",
    7: "黑将", 8: "黑士", 9: "黑象", 10: "黑车", 11: "黑马", 12: "黑炮", 13: "黑卒",
    14: "空",
}

EMPTY_IDX = 14
ROWS = 10
COLS = 9


def build_fen(predictions: list[int]) -> str:
    assert len(predictions) == ROWS * COLS, f"Expected {ROWS * COLS} predictions, got {len(predictions)}"

    rows = []
    for row in range(ROWS):
        row_str = ""
        empty_count = 0
        for col in range(COLS):
            idx = row * COLS + col
            pred = predictions[idx]

            if pred == EMPTY_IDX:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                fen_char = IDX_TO_FEN.get(pred, "")
                row_str += fen_char

        if empty_count > 0:
            row_str += str(empty_count)

        rows.append(row_str)

    return "/".join(rows)
