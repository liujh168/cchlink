import sys
sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np
from src.preprocess.board_detector import _detect_by_contour, _detect_by_hough_lines, _order_corners

test_paths = [
    r"i:\cchlink\data\raw\eval\test_001.png",
    r"i:\cchlink\data\raw\eval\test_004.png",
    r"i:\cchlink\data\raw\eval\test_009.png",
]

for tp in test_paths:
    img = cv2.imread(tp)
    if img is None:
        print(f"{tp}: not found")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    print(f"\n=== {tp.split(chr(92))[-1]} ({w}x{h}) ===")

    c_corners = _detect_by_contour(gray, h, w)
    print(f"Contour: {c_corners}")

    h_corners = _detect_by_hough_lines(gray, h, w)
    print(f"Hough: {h_corners}")

    vis = img.copy()
    if c_corners is not None:
        for pt in c_corners.astype(np.int32):
            cv2.circle(vis, tuple(pt), 8, (0, 255, 0), -1)
    if h_corners is not None:
        for pt in h_corners.astype(np.int32):
            cv2.circle(vis, tuple(pt), 8, (0, 0, 255), -1)

    out = tp.replace(".png", "_diag.jpg")
    cv2.imwrite(out, vis)
    print(f"Saved {out.split(chr(92))[-1]} (green=contour, red=hough)")