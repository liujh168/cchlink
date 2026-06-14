from src.recognition.split import group_split_indices


def test_group_split_has_no_board_leakage():
    groups = ["a"] * 90 + ["b"] * 90 + ["c"] * 90
    train, val = group_split_indices(groups, 0.34)

    assert set(groups[i] for i in train).isdisjoint(groups[i] for i in val)
    assert sorted(train + val) == list(range(len(groups)))
