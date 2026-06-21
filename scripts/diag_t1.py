import sys

sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np

tp = r"i:\cchlink\data\raw\eval\test_001.png"
img = cv2.imread(tp)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]

print(f"Image size: {w}x{h}")
print(
    f"Gray stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}, std={gray.std():.1f}"
)

blur = cv2.GaussianBlur(gray, (7, 7), 0)
_, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresh_50 = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

white_ratio = np.count_nonzero(thresh_otsu) / thresh_otsu.size
print(f"Otsu white ratio: {white_ratio:.3f}")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours found: {len(contours)}")
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    print(f"  contour[{i}]: area={area:.0f}, hull_area={hull_area:.0f}, ratio={area / (h * w):.3f}")

cv2.imwrite(r"i:\cchlink\data\raw\diag_t1_gray.jpg", gray)
cv2.imwrite(r"i:\cchlink\data\raw\diag_t1_otsu.jpg", thresh_otsu)
cv2.imwrite(r"i:\cchlink\data\raw\diag_t1_thresh50.jpg", thresh_50)
cv2.imwrite(r"i:\cchlink\data\raw\diag_t1_closed.jpg", closed)

vis = img.copy()
for c in contours:
    cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
cv2.imwrite(r"i:\cchlink\data\raw\diag_t1_contours.jpg", vis)

h, w = gray.shape[:2]
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
for th in [30, 50, 72, 100]:
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=th)
    cnt = 0 if lines is None else len(lines)
    print(f"HoughLines threshold={th}: {cnt} lines")

cv2.imwrite(r"i:\cchlink\data\raw\diag_t1_edges.jpg", edges)
print("Saved diagnostic images")
