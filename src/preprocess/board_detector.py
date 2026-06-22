from dataclasses import dataclass

import cv2
import numpy as np

from src.geometry import BOARD_HEIGHT, BOARD_WIDTH
from src.segmentation.grid_splitter import detect_grid


@dataclass(frozen=True)
class BoardDetection:
    """棋盘检测结果，包含四角坐标以及候选质量评分。"""

    corners: np.ndarray
    confidence: float
    grid_confidence: float


def detect_board_corners(image: np.ndarray) -> np.ndarray | None:
    """兼容旧接口，仅返回通过置信度校验的棋盘四角坐标。"""
    detection = detect_board(image)
    return detection.corners if detection is not None else None


def detect_board(image: np.ndarray, min_confidence: float = 0.22) -> BoardDetection | None:
    """从 RGB 图像中选择最可信的棋盘候选。

    轮廓法负责产生多个候选四边形，霍夫直线法作为补充候选来源。每个候选都会先
    透视校正，再结合内部网格响应和棋盘长宽比评分，避免直接选择最大轮廓而误判桌面、
    阴影或图片边框。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    candidates = _contour_candidates(gray, h, w)
    hough = _detect_by_hough_lines(gray, h, w)
    if hough is not None:
        candidates.append(hough)

    best = None
    for corners in candidates:
        detection = _score_candidate(image, corners)
        if best is None or detection.confidence > best.confidence:
            best = detection

    if best is None or best.confidence < min_confidence:
        return None
    return best


def _score_candidate(image: np.ndarray, corners: np.ndarray) -> BoardDetection:
    """根据校正后的网格质量和候选长宽比计算综合置信度。"""
    dst = np.float32(
        [
            [0, 0],
            [BOARD_WIDTH - 1, 0],
            [BOARD_WIDTH - 1, BOARD_HEIGHT - 1],
            [0, BOARD_HEIGHT - 1],
        ]
    )
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, transform, (BOARD_WIDTH, BOARD_HEIGHT))
    grid_confidence = detect_grid(warped).confidence

    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    ratio = ((top + bottom) / 2) / max((left + right) / 2, 1.0)
    ratio_score = float(np.exp(-abs(ratio - BOARD_WIDTH / BOARD_HEIGHT) * 2.0))
    image_area = max(float(image.shape[0] * image.shape[1]), 1.0)
    area_ratio = cv2.contourArea(corners.reshape(-1, 1, 2)) / image_area
    area_score = float(np.clip((area_ratio - 0.06) / 0.34, 0.0, 1.0))
    confidence = 0.66 * grid_confidence + 0.18 * ratio_score + 0.16 * area_score
    return BoardDetection(corners, float(confidence), float(grid_confidence))


def _contour_candidates(gray: np.ndarray, h: int, w: int) -> list[np.ndarray]:
    """从正向和反向二值图中提取并去重棋盘四边形候选。"""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    candidates = []
    image_area = h * w

    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, thresh = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            area = cv2.contourArea(contour)
            if not image_area * 0.04 <= area <= image_area * 0.98:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                corners = polygon.reshape(4, 2).astype(np.float32)
            else:
                corners = cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(contour)))
            candidates.append(_order_corners(corners))

    unique = []
    for candidate in candidates:
        if not any(np.mean(np.linalg.norm(candidate - other, axis=1)) < 8 for other in unique):
            unique.append(candidate)
    return unique[:25]


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
    A = np.array(
        [
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)],
        ]
    )
    b = np.array([rho1, rho2])
    try:
        pt = np.linalg.solve(A, b)
        return (pt[0], pt[1])
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
