import csv

import pytest

from scripts.audit_dataset import audit_dataset
from scripts.generate_grouped_data import generate_dataset


def test_standard_dataset_has_provenance_and_no_seed_leakage(tmp_path):
    dataset = tmp_path / "dataset"
    metadata = generate_dataset(
        dataset, num_boards=6, seed=1000, empty_keep_prob=1.0, scene_prob=1.0
    )
    eval_manifest = tmp_path / "eval.csv"
    eval_manifest.write_text("path,expected_fen,seed\nimage.png,9/9/9/9/9/9/9/9/9/9,9000\n")

    summary = audit_dataset(dataset, eval_manifest)

    assert metadata["num_boards"] == 6
    assert summary["groups"] == 6
    assert summary["scene_augmented"] == 6 * 90
    assert set(summary["styles"]) == {"classic", "plastic", "wood"}
    rows = list(csv.DictReader(open(dataset / "manifest.csv", encoding="utf-8")))
    assert all(row["source"] == "standard-v4" for row in rows)
    assert all(row["scene_augmented"] == "true" for row in rows)


def test_dataset_audit_rejects_eval_seed_leakage(tmp_path):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, num_boards=2, seed=1000, empty_keep_prob=1.0)
    eval_manifest = tmp_path / "eval.csv"
    eval_manifest.write_text("path,expected_fen,seed\nimage.png,9/9/9/9/9/9/9/9/9/9,1000\n")

    with pytest.raises(ValueError, match="leaked_seeds"):
        audit_dataset(dataset, eval_manifest)
