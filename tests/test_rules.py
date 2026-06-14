from src.fen.rules import legalize_predictions


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
