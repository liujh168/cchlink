import csv

from scripts.generate_standard_eval import generate_eval_set
from src.standard_board import STANDARD_INITIAL_FEN


def test_standard_eval_is_reproducible_and_uses_correct_initial_fen(tmp_path):
    output = tmp_path / "images"
    manifest = tmp_path / "manifest.csv"

    summary = generate_eval_set(output, manifest, count=6, seed=90000)

    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    assert summary["count"] == 6
    assert {row["style"] for row in rows} == {"classic", "plastic", "wood"}
    assert rows[0]["expected_fen"] == STANDARD_INITIAL_FEN
    assert all((output / row["path"].split("/")[-1]).is_file() for row in rows)
