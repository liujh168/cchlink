import sys

sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np

tp = r"i:\cchlink\data\raw\eval\test_001.png"
img = cv2.imread(tp)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 40, 120)

threshold = int(min(h, w) * 0.12)
print(f"Hough threshold: {threshold}")
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=threshold)
line_count = 0 if lines is None else len(lines)
print(f"Total lines: {line_count}")

if lines is not None:
    h_lines = []
    v_lines = []
    for i, line in enumerate(lines):
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        if i < 20:
            print(f"  line[{i}]: rho={rho:.1f}, theta={angle_deg:.1f}°")
        if angle_deg < 15 or angle_deg > 165:
            h_lines.append((rho, theta))
        elif 75 < angle_deg < 105:
            v_lines.append((rho, theta))

    print(f"Horizontal lines: {len(h_lines)}")
    print(f"Vertical lines: {len(v_lines)}")

    if len(h_lines) >= 5 and len(v_lines) >= 5:
        h_lines.sort(key=lambda x: x[0])
        v_lines.sort(key=lambda x: x[0])

        h_rhos = [lt[0] for lt in h_lines]
        v_rhos = [lt[0] for lt in v_lines]
        print(f"H rhos: {h_rhos[:10]}...{h_rhos[-3:]}")
        print(f"V rhos: {v_rhos[:10]}...{v_rhos[-3:]}")

        from src.preprocess.board_detector import _cluster_rhos

        h_clusters = _cluster_rhos(h_rhos)
        v_clusters = _cluster_rhos(v_rhos)
        print(f"H clusters: {h_clusters}")
        print(f"V clusters: {v_clusters}")
    else:
        print("Not enough h/v lines")
