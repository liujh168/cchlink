import os
import shutil
import random

src1 = r"i:\cchlink\data\pieces"
src2 = r"i:\cchlink\data\pieces_board"
dst = r"i:\cchlink\data\pieces_combined"

classes = ["红帅","红仕","红相","红俥","红马","红炮","红兵",
           "黑将","黑士","黑象","黑车","黑马","黑炮","黑卒","空"]

os.makedirs(dst, exist_ok=True)
for c in classes:
    os.makedirs(os.path.join(dst, c), exist_ok=True)

counters = {}
for c in classes:
    counters[c] = 0

    # standalone data
    src_dir = os.path.join(src1, c)
    if os.path.exists(src_dir):
        for f in sorted(os.listdir(src_dir)):
            if f.endswith(".png"):
                shutil.copy2(os.path.join(src_dir, f),
                            os.path.join(dst, c, f"{counters[c]:05d}.png"))
                counters[c] += 1

    # board data (limit empty cells)
    src_dir = os.path.join(src2, c)
    if os.path.exists(src_dir):
        files = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])
        if c == "空":
            random.Random(42).shuffle(files)
            files = files[:2000]
        for f in files:
            shutil.copy2(os.path.join(src_dir, f),
                        os.path.join(dst, c, f"{counters[c]:05d}.png"))
            counters[c] += 1

print("classes count:")
for c in classes:
    print(f"  {c}: {counters[c]}")
print(f"total: {sum(counters.values())}")