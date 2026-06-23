import csv
import random
from collections import Counter

import pytest

from scripts.audit_dataset import audit_dataset
from scripts.generate_grouped_data import (
    BLACK_GENERAL,
    RED_KING,
    choose_layout,
    generate_dataset,
    generate_random_opening,
    generate_sparse_endgame,
)
from src.standard_board import STANDARD_INITIAL_FEN, STANDARD_INITIAL_LAYOUT, layout_to_fen


def test_standard_dataset_has_provenance_and_no_seed_leakage(tmp_path):
    dataset = tmp_path / "dataset"
    metadata = generate_dataset(
        dataset, num_boards=10, seed=1000, empty_keep_prob=1.0, scene_prob=1.0
    )
    eval_manifest = tmp_path / "eval.csv"
    eval_manifest.write_text("path,expected_fen,seed\nimage.png,9/9/9/9/9/9/9/9/9/9,9000\n")

    summary = audit_dataset(dataset, eval_manifest)

    assert metadata["num_boards"] == 10
    assert summary["groups"] == 10
    assert summary["scene_augmented"] == 10 * 90
    assert summary["edge_augmented"] > 0
    assert summary["patch_scales"]
    assert set(summary["styles"]) == {
        "classic",
        "cloth_red",
        "light_cloth",
        "light_wood",
        "plastic",
        "screen",
        "wood",
    }
    rows = list(csv.DictReader(open(dataset / "manifest.csv", encoding="utf-8")))
    assert all(row["source"] == "standard-v9" for row in rows)
    assert all(row["scene_augmented"] == "true" for row in rows)
    assert all(row["patch_scale"] for row in rows)
    assert all(row["patch_shift_y"] for row in rows)
    assert all(row["patch_shift_x"] for row in rows)


def test_standard_dataset_edge_piece_augmentation_is_recorded(tmp_path):
    dataset = tmp_path / "dataset"
    generate_dataset(
        dataset,
        num_boards=1,
        seed=2000,
        empty_keep_prob=1.0,
        scene_prob=0.0,
        real_photo_prob=0.0,
        edge_aug_prob=1.0,
    )

    rows = list(csv.DictReader(open(dataset / "manifest.csv", encoding="utf-8")))
    edge_rows = [row for row in rows if row["edge_augmented"] == "true" and row["label"] != "空"]

    assert edge_rows
    edge_scales = {"0.60", "0.66", "0.72", "0.82", "0.92", "1.04"}
    assert all(row["patch_scale"] in edge_scales for row in edge_rows)
    assert any(row["patch_shift_y"] != "0" or row["patch_shift_x"] != "0" for row in edge_rows)


def test_dataset_audit_rejects_eval_seed_leakage(tmp_path):
    dataset = tmp_path / "dataset"
    generate_dataset(dataset, num_boards=2, seed=1000, empty_keep_prob=1.0)
    eval_manifest = tmp_path / "eval.csv"
    eval_manifest.write_text("path,expected_fen,seed\nimage.png,9/9/9/9/9/9/9/9/9/9,1000\n")

    with pytest.raises(ValueError, match="leaked_seeds"):
        audit_dataset(dataset, eval_manifest)


def test_v9_layout_cycle_matches_target_distribution():
    rng = random.Random(1234)
    counts = Counter(choose_layout(index, rng)[1] for index in range(25))

    assert counts == {
        "initial": 2,
        "empty": 2,
        "opening": 5,
        "midgame": 7,
        "sparse_endgame": 9,
    }


def test_v8_opening_layout_moves_from_initial_position():
    layout = generate_random_opening(random.Random(4321))

    assert layout != STANDARD_INITIAL_LAYOUT
    assert layout_to_fen(layout) != STANDARD_INITIAL_FEN
    assert any(piece == BLACK_GENERAL for row in layout for piece in row)
    assert any(piece == RED_KING for row in layout for piece in row)


def test_v8_sparse_endgame_keeps_kings_and_piece_count_range():
    layout = generate_sparse_endgame(random.Random(9876))
    pieces = [piece for row in layout for piece in row if piece is not None]

    assert BLACK_GENERAL in pieces
    assert RED_KING in pieces
    assert 3 <= len(pieces) <= 12
