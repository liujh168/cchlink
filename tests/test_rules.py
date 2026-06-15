from src.fen.fen_builder import build_fen
from src.fen.rules import (
    detect_orientation,
    legalize_predictions,
    normalize_orientation,
    validate_position,
)


def test_rules_remove_low_confidence_extra_and_out_of_palace_kings():
    predictions = [14] * 90
    confidences = [0.99] * 90
    predictions[0] = 0
    confidences[0] = 0.2
    predictions[76] = 0
    confidences[76] = 0.9
    predictions[77] = 0
    confidences[77] = 0.3

    result = legalize_predictions(predictions, confidences)

    assert result[0] == 14
    assert result[76] == 0
    assert result[77] == 14


def test_rules_support_red_side_on_top():
    predictions = [14] * 90
    confidences = [0.99] * 90
    predictions[4] = 0
    predictions[85] = 7
    predictions[9] = 6
    predictions[80] = 13

    assert legalize_predictions(predictions, confidences)[4] == 0


def test_orientation_normalizes_red_top_and_red_bottom_to_same_board():
    red_bottom = [14] * 90
    red_bottom[4] = 7
    red_bottom[85] = 0
    red_top = list(reversed(red_bottom))
    confidences = [0.9] * 90

    assert detect_orientation(red_bottom) == "red_bottom"
    assert detect_orientation(red_top) == "red_top"
    normalized_top = normalize_orientation(red_top, confidences, "red_top")[0]
    normalized_bottom = normalize_orientation(red_bottom, confidences, "red_bottom")[0]
    assert build_fen(normalized_top) == build_fen(normalized_bottom)


def test_orientation_unknown_for_empty_board():
    assert detect_orientation([14] * 90) == "unknown"


def test_static_validation_reports_all_supported_rule_families():
    board = [14] * 90
    board[0] = 0
    board[4] = 7
    board[10] = 1
    board[11] = 2
    board[55] = 6
    board[22] = 3
    board[23] = 3
    board[24] = 3

    codes = {warning.code for warning in validate_position(board, "unknown")}

    assert "orientation_unknown" in codes
    assert "king_outside_palace" in codes
    assert "red_advisor_illegal" in codes
    assert "red_elephant_illegal" in codes
    assert "red_pawn_illegal" in codes
    assert "piece_count_exceeded" in codes


def test_static_validation_reports_missing_kings_and_facing_kings():
    empty_codes = {warning.code for warning in validate_position([14] * 90, "red_bottom")}
    assert {"king_count", "general_count"} <= empty_codes

    board = [14] * 90
    board[4] = 7
    board[85] = 0
    codes = {warning.code for warning in validate_position(board, "red_bottom")}
    assert "kings_facing" in codes


def test_high_confidence_illegal_prediction_is_preserved_and_warned():
    board = [14] * 90
    board[0] = 0
    confidences = [0.99] * 90

    legalized = legalize_predictions(board, confidences)
    codes = {warning.code for warning in validate_position(legalized, "red_bottom")}

    assert legalized[0] == 0
    assert "king_outside_palace" in codes
