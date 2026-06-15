import pytest

from src.fen.fen_builder import build_fen
from src.recognition.dataset import CLASS_TO_IDX
from src.standard_board import STANDARD_INITIAL_FEN, STANDARD_INITIAL_LAYOUT


def test_build_initial_position_fen():
    board = [
        CLASS_TO_IDX[piece] if piece else CLASS_TO_IDX["空"]
        for row in STANDARD_INITIAL_LAYOUT
        for piece in row
    ]
    assert build_fen(board) == STANDARD_INITIAL_FEN


def test_build_fen_requires_90_predictions():
    with pytest.raises(ValueError):
        build_fen([14] * 89)
