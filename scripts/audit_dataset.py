import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="审计按棋盘分组的数据集清单")
    parser.add_argument("dataset", help="包含 manifest.csv 的数据集目录")
    args = parser.parse_args()
    root = Path(args.dataset)
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))
    hashes = defaultdict(list)
    labels = Counter()
    groups = Counter()

    for row in rows:
        path = root / row["path"]
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(row)
        labels[row["label"]] += 1
        groups[row["group"]] += 1

    duplicates = [items for items in hashes.values() if len(items) > 1]
    cross_group = [items for items in duplicates if len({item["group"] for item in items}) > 1]
    print(f"samples={len(rows)} groups={len(groups)} labels={len(labels)}")
    print(f"duplicate_hash_groups={len(duplicates)} cross_group_duplicates={len(cross_group)}")
    print(f"group_size_min={min(groups.values())} group_size_max={max(groups.values())}")
    for label, count in sorted(labels.items()):
        print(f"{label}: {count}")
    if cross_group:
        raise SystemExit("发现跨棋盘分组的完全重复图片")


if __name__ == "__main__":
    main()
