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
        }
    ]
