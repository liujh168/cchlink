from scripts.diagnose_eval_errors import compare_fens, fen_cells


def test_fen_cells_expands_empty_rows():
    cells = fen_cells("9/9/9/9/9/9/9/9/9/9")

    assert len(cells) == 90
    assert set(cells) == {""}


def test_compare_fens_reports_position_and_piece_names():
    errors = compare_fens(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
        "rnbakabn1/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
    )

    assert errors == [
        {
            "row": 0,
            "col": 8,
            "region": "edge",
            "expected": "r",
            "actual": "",
            "expected_name": "黑车",
            "actual_name": "空",
            "kind": "piece_to_empty",
        }
    ]


def test_compare_fens_classifies_error_kinds():
    errors = compare_fens(
        "9/9/9/9/9/9/9/9/9/9",
        "N8/9/9/9/9/9/9/9/9/9",
    )

    assert errors[0]["kind"] == "empty_to_piece"

    errors = compare_fens(
        "N8/9/9/9/9/9/9/9/9/9",
        "n8/9/9/9/9/9/9/9/9/9",
    )

    assert errors[0]["kind"] == "color_confusion"
