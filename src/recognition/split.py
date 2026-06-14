from __future__ import annotations

import random
from collections import defaultdict


def group_split_indices(groups: list[str], val_fraction: float, seed: int = 42):
    """按棋盘分组拆分训练集和验证集，避免同盘样本跨集合泄漏。

    每个 group 通常代表一张完整棋盘。拆分时整个分组只会进入训练集或验证集之一，
    从而使验证指标更接近模型面对未见棋盘时的真实泛化能力。
    """
    grouped_indices = defaultdict(list)
    for index, group in enumerate(groups):
        grouped_indices[group].append(index)

    group_names = sorted(grouped_indices)
    random.Random(seed).shuffle(group_names)
    target = round(len(groups) * val_fraction)
    val_indices = []
    train_indices = []
    for group in group_names:
        destination = val_indices if len(val_indices) < target else train_indices
        destination.extend(grouped_indices[group])
    return train_indices, val_indices
