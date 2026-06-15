import sys
sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np
from src.preprocess.board_detector import _compute_grid_homography, _cluster_rhos, _hough_intersection

test_id = '004'
img_path = rf'i:\cchlink\data\raw\eval\test_{test_id}.png'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

for canny_low, canny_high, hough_thresh_factor in [(40, 120, 0.15), (40, 120, 0.12), (50, 150, 0.10), (30, 90, 0.08)]:
    edges = cv2.Canny(blurred, canny_low, canny_high)
    threshold = int(min(h, w) * hough_thresh_factor)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=threshold)
    if lines is None:
        continue

    h_rhos = []
    v_rhos = []
    _h_angles = []
    _v_angles = []

    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        if angle_deg < 15 or angle_deg > 165:
            h_rhos.append(rho)
            _h_angles.append(angle_deg)
        elif 75 < angle_deg < 105:
            v_rhos.append(rho)
            _v_angles.append(angle_deg)

    h_rhos.sort()
    v_rhos.sort()

    print(f'  canny={canny_low},{canny_high} thresh={threshold} h_lines={len(h_rhos)} v_lines={len(v_rhos)}')
    print(f'  h_rhos: {[round(r,1) for r in h_rhos[:10]]}...')
    print(f'  v_rhos: {[round(r,1) for r in v_rhos[:10]]}...')
    print(f'  h_angles: {[round(a,1) for a in _h_angles[:5]]}...')
    print(f'  v_angles: {[round(a,1) for a in _v_angles[:5]]}...')

    h_clusters = _cluster_rhos(h_rhos)
    v_clusters = _cluster_rhos(v_rhos)
    print(f'  h_clusters ({len(h_clusters)}): {h_clusters}')
    print(f'  v_clusters ({len(v_clusters)}): {v_clusters}')

    theta_h = np.mean([np.deg2rad(d) for d in _h_angles]) if _h_angles else 0.0
    theta_v = np.mean([np.deg2rad(d) for d in _v_angles]) if _v_angles else np.pi / 2
    print(f'  theta_h={np.degrees(theta_h):.1f} theta_v={np.degrees(theta_v):.1f}')

    if len(h_clusters) >= 3 and len(v_clusters) >= 3:
        h_min, h_max = min(h_clusters), max(h_clusters)
        v_min, v_max = min(v_clusters), max(v_clusters)
        print(f'  h_range={h_max-h_min:.0f} v_range={v_max-v_min:.0f}')

        src_pts = []
        dst_pts = []
        for rho_h in h_clusters:
            col = (rho_h - h_min) / max(h_max - h_min, 1) * 449
            for rho_v in v_clusters:
                row = (rho_v - v_min) / max(v_max - v_min, 1) * 499
                sx, sy = _hough_intersection(rho_h, theta_h, rho_v, theta_v)
                print(f'    rho_h={rho_h:.0f} rho_v={rho_v:.0f} -> col={col:.1f} row={row:.1f} src=({sx:.0f},{sy:.0f})')
                if 0 <= sx < w and 0 <= sy < h:
                    src_pts.append([sx, sy])
                    dst_pts.append([col, row])

        print(f'  total valid intersections: {len(src_pts)}')
        break