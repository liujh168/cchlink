import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_STANDARD_FIELDS = {
    "path",
    "label",
    "group",
    "style",
    "layout_id",
    "layout_type",
    "seed",
    "orientation",
    "source",
    "fen",
}
STANDARD_DATASET_SOURCES = {
    "standard-v2",
    "standard-v3",
    "standard-v4",
    "standard-v5",
    "standard-v6",
    "standard-v7",
    "standard-v8",
    "standard-v9",
}


def audit_dataset(root: Path, against_manifest: Path | None = None) -> dict:
    manifest = root / "manifest.csv"
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    if not rows:
        raise ValueError("数据集清单为空")
    fieldnames = set(rows[0])
    missing_fields = REQUIRED_STANDARD_FIELDS - fieldnames
    if missing_fields:
        raise ValueError(f"清单缺少标准字段: {sorted(missing_fields)}")

    hashes = defaultdict(list)
    labels = Counter()
    groups = Counter()
    styles = Counter()
    layout_types = Counter()
    patch_scales = Counter()
    missing_files = []
    seeds = set()
    group_metadata = {}
    for row in rows:
        if row["source"] not in STANDARD_DATASET_SOURCES:
            raise ValueError(f"未知标准数据源: {row['source']}")
        if (
            row["source"]
            in {
                "standard-v3",
                "standard-v4",
                "standard-v5",
                "standard-v6",
                "standard-v7",
                "standard-v8",
                "standard-v9",
            }
            and "scene_augmented" not in fieldnames
        ):
            raise ValueError(f"{row['source']} 清单缺少 scene_augmented 字段")
        if row["source"] in {
            "standard-v5",
            "standard-v6",
            "standard-v7",
            "standard-v8",
            "standard-v9",
        }:
            patch_fields = {"patch_scale", "patch_shift_y", "patch_shift_x", "edge_augmented"}
            missing_patch_fields = patch_fields - fieldnames
            if missing_patch_fields:
                raise ValueError(
                    f"{row['source']} 清单缺少字段: {sorted(missing_patch_fields)}"
                )
        path = root / row["path"]
        if not path.is_file():
            missing_files.append(row["path"])
            continue
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(row)
        labels[row["label"]] += 1
        groups[row["group"]] += 1
        styles[row["style"]] += 1
        layout_types[row["layout_type"]] += 1
        if row.get("patch_scale"):
            patch_scales[row["patch_scale"]] += 1
        seeds.add(int(row["seed"]))
        metadata = (
            row["style"],
            row["layout_id"],
            row["layout_type"],
            row["seed"],
            row["orientation"],
            row["fen"],
        )
        previous = group_metadata.setdefault(row["group"], metadata)
        if previous != metadata:
            raise ValueError(f"分组元数据不一致: {row['group']}")

    duplicates = [items for items in hashes.values() if len(items) > 1]
    cross_group = [items for items in duplicates if len({item["group"] for item in items}) > 1]
    leaked_seeds = []
    if against_manifest is not None:
        against = list(csv.DictReader(open(against_manifest, encoding="utf-8")))
        eval_seeds = {int(row["seed"]) for row in against if row.get("seed")}
        leaked_seeds = sorted(seeds & eval_seeds)

    summary = {
        "samples": len(rows),
        "groups": len(groups),
        "labels": dict(sorted(labels.items())),
        "styles": dict(sorted(styles.items())),
        "layout_types": dict(sorted(layout_types.items())),
        "scene_augmented": sum(row.get("scene_augmented") == "true" for row in rows),
        "edge_augmented": sum(row.get("edge_augmented") == "true" for row in rows),
        "patch_scales": dict(sorted(patch_scales.items())),
        "duplicate_hash_groups": len(duplicates),
        "cross_group_duplicates": len(cross_group),
        "missing_files": missing_files,
        "leaked_seeds": leaked_seeds,
        "group_size_min": min(groups.values()),
        "group_size_max": max(groups.values()),
    }
    if missing_files or cross_group or leaked_seeds:
        raise ValueError(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description="审计标准分组数据集清单")
    parser.add_argument("dataset", help="包含 manifest.csv 的数据集目录")
    parser.add_argument("--against-manifest", help="检查与评估清单的随机种子泄漏")
    args = parser.parse_args()
    summary = audit_dataset(
        Path(args.dataset),
        Path(args.against_manifest) if args.against_manifest else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
