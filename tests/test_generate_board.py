import numpy as np
from PIL import Image, ImageDraw

from scripts.generate_board import (
    BOARD_STYLES,
    INITIAL_LAYOUT,
    _draw_position_marker,
    _grid_point,
    render_board,
)
from src.standard_board import STANDARD_INITIAL_FEN, layout_to_fen


def test_initial_layout_uses_standard_piece_rows():
    assert INITIAL_LAYOUT[1] == [None] * 9
    assert INITIAL_LAYOUT[2][1] == INITIAL_LAYOUT[2][7] == "黑炮"
    assert [INITIAL_LAYOUT[3][col] for col in (0, 2, 4, 6, 8)] == ["黑卒"] * 5
    assert [INITIAL_LAYOUT[6][col] for col in (0, 2, 4, 6, 8)] == ["红兵"] * 5
    assert INITIAL_LAYOUT[7][1] == INITIAL_LAYOUT[7][7] == "红炮"
    assert INITIAL_LAYOUT[8] == [None] * 9
    assert layout_to_fen(INITIAL_LAYOUT) == STANDARD_INITIAL_FEN


def test_scaled_grid_points_match_piece_centers():
    assert _grid_point(0, 0, 3) == (75, 75)
    assert _grid_point(9, 8, 3) == (1275, 1425)


def test_board_styles_render_at_high_resolution():
    for style in ("wood", "plastic"):
        image = np.asarray(render_board(INITIAL_LAYOUT, style=style, scale=3))
        assert image.shape == (1500, 1350, 3)


def test_edge_position_marker_only_points_inward():
    background = BOARD_STYLES["wood"]["background"]
    color = BOARD_STYLES["wood"]["line"]
    image = Image.new("RGB", (100, 100), background)
    draw = ImageDraw.Draw(image)

    _draw_position_marker(draw, row=0, col=0, scale=1, color=color)

    array = np.asarray(image)
    _, x = _grid_point(0, 0)
    assert np.any(array[:, x + 6 : x + 12] != background)
    assert np.all(array[:, : x - 5] == background)
