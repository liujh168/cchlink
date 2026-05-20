import cv2
import numpy as np


def detect_board_corners(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    corners = _detect_by_contour(gray, h, w)
    if corners is not None:
        return corners

    corners = _detect_by_hough_lines(gray, h, w)
    if corners is not None:
        return corners

    return None


def _detect_by_hough_lines(gray: np.ndarray, h: int, w: int) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    threshold = int(min(h, w) * 0.12)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=threshold)
    if lines is None:
        return None

    h_lines = []
    v_lines = []

    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        if angle_deg < 15 or angle_deg > 165:
            h_lines.append((rho, theta))
        elif 75 < angle_deg < 105:
            v_lines.append((rho, theta))

    if len(h_lines) < 5 or len(v_lines) < 5:
        return None

    h_lines.sort(key=lambda x: x[0])
    v_lines.sort(key=lambda x: x[0])

    h_rhos = [lt[0] for lt in h_lines]
    v_rhos = [lt[0] for lt in v_lines]
    h_clusters = _cluster_rhos(h_rhos)
    v_clusters = _cluster_rhos(v_rhos)

    if len(h_clusters) < 5 or len(v_clusters) < 5:
        return None

    outer_h = (h_clusters[0], h_clusters[-1])
    outer_v = (v_clusters[0], v_clusters[-1])

    avg_theta_h = np.mean([lt[1] for lt in h_lines])
    avg_theta_v = np.mean([lt[1] for lt in v_lines])

    corners = []
    for rho_h in outer_h:
        for rho_v in outer_v:
            pt = _hough_intersection(rho_h, avg_theta_h, rho_v, avg_theta_v)
            corners.append(pt)

    corners = np.array(corners, dtype=np.float32)
    corners = _order_corners(corners)

    for pt in corners:
        if pt[0] < -80 or pt[0] > w + 80 or pt[1] < -80 or pt[1] > h + 80:
            return None

    area = cv2.contourArea(corners.reshape(-1, 1, 2))
    if area < w * h * 0.08:
        return None

    return corners


def _hough_intersection(rho1, theta1, rho2, theta2):
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)],
    ])
    b = np.array([rho1, rho2])
    try:
        pt = np.linalg.solve(A, b)
        return (pt[1], pt[0])
    except np.linalg.LinAlgError:
        return (0, 0)


def _detect_by_contour(gray: np.ndarray, h: int, w: int) -> np.ndarray | None:
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.count_nonzero(thresh) / thresh.size
    if white_ratio < 0.05 or white_ratio > 0.95:
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    image_area = h * w

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = [c for c in contours if cv2.contourArea(c) > image_area * 0.005]
    contours = [c for c in contours if cv2.contourArea(c) < image_area * 0.95]

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    if hull_area < image_area * 0.02:
        return None

    rect = cv2.minAreaRect(hull)
    corners = cv2.boxPoints(rect)
    corners = corners.astype(np.float32)
    corners = _order_corners(corners)

    for pt in corners:
        if pt[0] < -80 or pt[0] > w + 80 or pt[1] < -80 or pt[1] > h + 80:
            return None

    board_area = cv2.contourArea(corners.reshape(-1, 1, 2))
    if board_area < image_area * 0.05:
        return None

    return corners


def _cluster_rhos(rhos: list[float]) -> list[int]:
    if len(rhos) < 2:
        return [int(r) for r in rhos]

    clusters = []
    current = [rhos[0]]
    for rho in rhos[1:]:
        if abs(rho - current[-1]) < 10:
            current.append(rho)
        else:
            clusters.append(int(np.mean(current)))
            current = [rho]
    clusters.append(int(np.mean(current)))
    return clusters


def _order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect