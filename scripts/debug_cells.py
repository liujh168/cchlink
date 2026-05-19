import sys
sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np
import os
import random

# Check confusion classes
src_dir = r"i:\cchlink\data\pieces_board"
classes = ["红俥", "红炮", "红相"]

for c in classes:
    path = os.path.join(src_dir, c)
    files = os.listdir(path)
    rd = random.Random(1)
    sample = rd.sample(files, min(4, len(files)))
    print(f"\n=== {c} (n={len(files)}) ===")
    for f in sample:
        img = cv2.imread(os.path.join(path, f))
        if img is not None:
            h, w = img.shape[:2]
            # Show center region
            center = img[h//4:3*h//4, w//4:3*w//4]
            b, g, r = center[:,:,0].mean(), center[:,:,1].mean(), center[:,:,2].mean()
            print(f"  {f}: shape={img.shape}, center mean=(B={b:.0f},G={g:.0f},R={r:.0f})")

# Also check row 9 corner debug images
for c in [0, 8]:
    path = rf"i:\cchlink\data\raw\debug_cell_r9_c{c}.jpg"
    img = cv2.imread(path)
    if img is not None:
        b, g, r = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
        print(f"\ndebug_cell_r9_c{c}: shape={img.shape}, mean=(B={b:.0f},G={g:.0f},R={r:.0f})")