import os
import random
import shutil

random.seed(42)

src_dirs = [
    r"i:\cchlink\data\pieces_combined_v3",
    r"i:\cchlink\data\pieces_boards_v4",
]
dst = r"i:\cchlink\data\pieces_combined_v7"

classes = [
    "红帅",
    "红仕",
    "红相",
    "红俥",
    "红马",
    "红炮",
    "红兵",
    "黑将",
    "黑士",
    "黑象",
    "黑车",
    "黑马",
    "黑炮",
    "黑卒",
    "空",
]

MAX_EMPTY = 10000

for c in classes:
    os.makedirs(os.path.join(dst, c), exist_ok=True)

counters = {c: 0 for c in classes}

for src in src_dirs:
    if not os.path.exists(src):
        print(f"跳过不存在的目录: {src}")
        continue
    for c in classes:
        src_class_dir = os.path.join(src, c)
        if not os.path.exists(src_class_dir):
            continue
        files = sorted(os.listdir(src_class_dir))
        for f in files:
            if c == "空" and counters[c] >= MAX_EMPTY:
                break
            src_path = os.path.join(src_class_dir, f)
            dst_path = os.path.join(dst, c, f"{counters[c]:06d}.png")
            shutil.copy2(src_path, dst_path)
            counters[c] += 1

print("各类别统计:")
for c in classes:
    print(f"  {c}: {counters[c]} 张")
print(f"总计: {sum(counters.values())} 张")
