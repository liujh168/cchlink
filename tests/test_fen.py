import pytest

from src.fen.fen_builder import build_fen


def test_build_initial_position_fen():
    board = [
        10,
        11,
        9,
        8,
        7,
        8,
        9,
        11,
        10,
        14,
        12,
        14,
        14,
        14,
        14,
        14,
        12,
        14,
        13,
        14,
        13,
        14,
        13,
        14,
        13,
        14,
        13,
        *([14] * 36),
        6,
        14,
        6,
        14,
        6,
        14,
        6,
        14,
        6,
        14,
        5,
        14,
        14,
        14,
        14,
        14,
        5,
        14,
        3,
        4,
        2,
        1,
        0,
        1,
        2,
        4,
        3,
    ]
    assert build_fen(board) == ("rnbakabnr/1c5c1/p1p1p1p1p/9/9/9/9/P1P1P1P1P/1C5C1/RNBAKABNR")


def test_build_fen_requires_90_predictions():
    with pytest.raises(ValueError):
        build_fen([14] * 89)
