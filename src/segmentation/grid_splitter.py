from dataclasses import dataclass

import cv2
import numpy as np

from src.geometry import COLS, extract_intersection_patches, ideal_grid_positions


@dataclass(frozen=True)
class GridDetection:
    """校正棋盘中的交点位置及其整体可信度。"""

    row_positions: list[int]
    col_positions: list[int]
    confidence: float


def split_board(board_image: np.ndarray) -> list[np.ndarray]:
    detection = detect_grid(board_image)
    return extract_intersection_patches(
        board_image, detection.row_positions, detection.col_positions
    )


def split_board_with_positions(board_image: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    cells = split_board(board_image)
    return [(i // COLS, i % COLS, cell) for i, cell in enumerate(cells)]


def detect_grid(board_image: np.ndarray) -> GridDetection:
    """定位 10 行、9 列棋盘交点，并返回保守的整体置信度。"""
    gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    ideal_rows, ideal_cols = ideal_grid_positions(w, h)
    rows, row_score = _snap_intersections(gray, axis="h", ideal=ideal_rows)
    cols, col_score = _snap_intersections(gray, axis="v", ideal=ideal_cols)
    rows = _regularize_intersections(rows, ideal_rows)
    cols = _regularize_intersections(cols, ideal_cols)
    return GridDetection(rows, cols, float(min(row_score, col_score)))


def _regularize_intersections(positions: list[int], ideal: list[int]) -> list[int]:
    """用全局中位偏移约束 9x10 交点，避免局部棋子边缘抢走网格峰值。

    真实照片中外圈棋子常贴近校正棋盘边缘，Sobel 投影的局部最大值容易落在棋子
    外沿而不是棋盘线。透视校正后的棋盘仍应接近等距网格，因此保留整体平移量，
    再使用理想等距位置，可以降低外圈裁剪偏移。
    """
    if len(positions) != len(ideal):
        return positions
    offsets = np.asarray(positions, dtype=np.float32) - np.asarray(ideal, dtype=np.float32)
    offset = float(np.median(offsets))
    return [int(round(value + offset)) for value in ideal]


def _detect_grid_lines(gray: np.ndarray, shape: tuple) -> tuple[list[int], list[int]]:
    h, w = shape
    rows, cols = ideal_grid_positions(w, h)
    h_lines, _ = _snap_intersections(gray, axis="h", ideal=rows)
    v_lines, _ = _snap_intersections(gray, axis="v", ideal=cols)
    return h_lines, v_lines


def _snap_intersections(gray: np.ndarray, axis: str, ideal: list[int]) -> tuple[list[int], float]:
    """将理想交点吸附到附近的 Sobel 边缘峰值。

    置信度同时考虑峰值相对背景的强度、整条投影曲线的对比度以及相邻交点间距的
    规则程度。最终取横纵两个方向的较低值，使任一方向定位不可靠时能够触发拒识。
    """
    if axis == "h":
        edge = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        profile = edge.mean(axis=1)
    else:
        edge = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        profile = edge.mean(axis=0)

    profile = np.convolve(profile, np.ones(5, dtype=np.float32) / 5, mode="same")
    spacing = float(np.median(np.diff(ideal)))
    search_half = max(3, int(round(spacing * 0.22)))
    positions = []
    strengths = []
    baseline = float(np.percentile(profile, 55)) + 1e-6

    for center in ideal:
        lo = max(0, center - search_half)
        hi = min(len(profile), center + search_half + 1)
        window = profile[lo:hi]
        best = lo + int(np.argmax(window))
        positions.append(best)
        strengths.append(float(profile[best]) / baseline)

    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + 1)

    spacing_error = np.std(np.diff(positions)) / max(spacing, 1.0)
    strength_score = np.clip((np.median(strengths) - 1.0) / 2.5, 0.0, 1.0)
    p50 = float(np.percentile(profile, 50))
    p90 = float(np.percentile(profile, 90))
    contrast_score = np.clip(((p90 / (p50 + 1e-6)) - 1.0) / 3.0, 0.0, 1.0)
    signal_score = max(strength_score, contrast_score * 0.75)
    regularity_score = np.clip(1.0 - spacing_error / 0.22, 0.0, 1.0)
    confidence = signal_score * (0.65 + 0.35 * regularity_score)
    return positions, float(confidence)


def _snap_grid(gray: np.ndarray, axis: str, count: int, size: int) -> list[int]:
    if axis == "h":
        edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        profile = edge.mean(axis=1)
    else:
        edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        profile = edge.mean(axis=0)

    profile = np.convolve(profile, np.ones(5) / 5, mode="same")

    spacing = float(size) / (count - 1)
    best_offset = 0
    best_score = -1.0
    search_range = max(1, int(spacing * 0.3))

    for offset in range(-search_range, search_range + 1):
        score = 0.0
        for i in range(count):
            pos = int(offset + i * spacing)
            if 0 <= pos < len(profile):
                score += profile[pos]
        if score > best_score:
            best_score = score
            best_offset = offset

    lines = []
    for i in range(count):
        center = int(best_offset + i * spacing)
        lo = max(0, center - 2)
        hi = min(len(profile) - 1, center + 2)
        if lo >= hi:
            lines.append(center)
            continue
        window = profile[lo : hi + 1]
        best = int(np.argmax(window)) + lo
        lines.append(best)

    lines = sorted(set(lines))
    if not lines:
        return [int(i * spacing) for i in range(count)]

    if lines[0] > max(3, size // 20):
        lines.insert(0, 0)
    if lines[-1] < size - max(3, size // 20):
        lines.append(size - 1)

    while len(lines) < count:
        max_gap = 0
        max_idx = 0
        for i in range(len(lines) - 1):
            gap = lines[i + 1] - lines[i]
            if gap > max_gap:
                max_gap = gap
                max_idx = i
        lines.insert(max_idx + 1, (lines[max_idx] + lines[max_idx + 1]) // 2)

    if len(lines) > count:
        step = (len(lines) - 1) / (count - 1)
        selected = []
        for i in range(count):
            idx = min(len(lines) - 1, int(round(i * step)))
            selected.append(lines[idx])
        lines = selected

    return lines[:count]


def _detect_expected(gray: np.ndarray, axis: str, expected_count: int, size: int) -> list[int]:
    if axis == "h":
        edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        profile = edge.mean(axis=1)
    else:
        edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        profile = edge.mean(axis=0)

    profile = np.convolve(profile, np.ones(5) / 5, mode="same")

    spacing = size / (expected_count - 1)
    search_half = max(2, int(spacing * 0.25))

    lines = []
    for i in range(expected_count):
        center = int(i * spacing)
        lo = max(0, center - search_half)
        hi = min(len(profile) - 1, center + search_half)
        if lo >= hi:
            lines.append(center)
            continue
        window = profile[lo : hi + 1]
        best = int(np.argmax(window)) + lo
        lines.append(best)

    lines.sort()
    for i in range(1, len(lines)):
        if lines[i] <= lines[i - 1]:
            lines[i] = lines[i - 1] + 1

    return lines


def _ensure_boundaries(lines: list[int], size: int) -> list[int]:
    if not lines:
        return lines
    lines = sorted(lines)
    if lines[0] > max(3, size // 20):
        lines.insert(0, 0)
    if lines[-1] < size - max(3, size // 20):
        lines.append(size - 1)
    return lines


def _detect_by_sobel(gray: np.ndarray, axis: str, expected: int) -> list[int]:
    if axis == "h":
        edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        profile = edge.mean(axis=1)
    else:
        edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        profile = edge.mean(axis=0)

    profile = np.convolve(profile, np.ones(5) / 5, mode="same")

    min_dist = max(5, len(profile) // (expected * 2))
    threshold = np.mean(profile) * 1.2

    peaks = []
    for i in range(min_dist, len(profile) - min_dist):
        if profile[i] <= threshold:
            continue
        is_peak = True
        for j in range(i - min_dist, i + min_dist + 1):
            if j != i and j < len(profile) and profile[j] > profile[i]:
                is_peak = False
                break
        if is_peak:
            peaks.append(i)

    peaks.sort()
    return peaks


def _detect_by_hough(gray: np.ndarray, h: int, w: int) -> tuple[list[int], list[int]]:
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(min(h, w) * 0.06))

    h_clusters = []
    v_clusters = []

    if lines is not None:
        h_rhos = []
        v_rhos = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            if angle_deg < 10 or angle_deg > 170:
                h_rhos.append(abs(rho))
            elif 80 < angle_deg < 100:
                v_rhos.append(abs(rho))

        h_rhos.sort()
        v_rhos.sort()
        h_clusters = _cluster_rhos(h_rhos)
        v_clusters = _cluster_rhos(v_rhos)

    return h_clusters, v_clusters


def _cluster_rhos(rhos: list[float]) -> list[int]:
    if len(rhos) < 2:
        return [int(r) for r in rhos]

    clusters = []
    current = [rhos[0]]
    for rho in rhos[1:]:
        if rho - current[-1] < 8:
            current.append(rho)
        else:
            clusters.append(int(np.mean(current)))
            current = [rho]
    clusters.append(int(np.mean(current)))
    return clusters


def _fill_lines(lines: list[int], expected: int, size: int) -> list[int]:
    if len(lines) < 2:
        return lines

    first = lines[0]
    last = lines[-1]
    spacing = (last - first) / (expected - 1)
    return [int(first + i * spacing) for i in range(expected)]


def _trim_lines(lines: list[int], expected: int) -> list[int]:
    if len(lines) <= expected:
        return lines

    step = (len(lines) - 1) / (expected - 1)
    result = []
    for i in range(expected):
        idx = int(round(i * step))
        idx = min(idx, len(lines) - 1)
        result.append(lines[idx])
    return result
