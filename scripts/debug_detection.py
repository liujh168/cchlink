import sys
sys.path.insert(0, r"i:\cchlink")

import cv2
import numpy as np

image_path = r"i:\cchlink\data\raw\initial_00.jpg"
image = cv2.imread(image_path)
print(f"Image shape: {image.shape}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
cv2.imwrite(r"i:\cchlink\data\raw\debug_thresh.jpg", thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Threshold contours: {len(contours)}")

if not contours:
    print("No contours found from threshold!")
    sys.exit(1)

largest = max(contours, key=cv2.contourArea)
area = cv2.contourArea(largest)
print(f"Largest contour area: {area:.0f}")

peri = cv2.arcLength(largest, True)
approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
print(f"approxPolyDP vertices: {len(approx)}")

if len(approx) < 4:
    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    print(f"After convexHull, vertices: {len(approx)}")

if len(approx) >= 4:
    if len(approx) > 4:
        rect = cv2.minAreaRect(largest)
        corners = cv2.boxPoints(rect)
        print(f"Using minAreaRect, corners shape: {corners.shape}")
    else:
        corners = approx.reshape(4, 2).astype(np.float32)
        print(f"Using approxPolyDP, corners:\n{corners}")

    corners = corners.astype(np.float32)

    s = corners.sum(axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = corners[np.argmin(s)]
    ordered[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    ordered[1] = corners[np.argmin(diff)]
    ordered[3] = corners[np.argmax(diff)]

    print(f"\nOrdered corners:\n{ordered}")

    debug = image.copy()
    for pt in ordered:
        cv2.circle(debug, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
    cv2.drawContours(debug, [ordered.astype(np.int32)], 0, (0, 255, 0), 2)
    cv2.imwrite(r"i:\cchlink\data\raw\debug_corners.jpg", debug)
    print("Saved debug_corners.jpg")
else:
    print(f"Still cannot get 4 corners, vertices={len(approx)}")