import cv2
import numpy as np


COLS = 9
ROWS = 10


def split_board(board_image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = _detect_grid_lines(gray, board_image.shape[:2])

    cells = []
    for row in range(min(ROWS, len(h_lines) - 1)):
        y1 = max(0, h_lines[row])
        y2 = min(board_image.shape[0], h_lines[row + 1])
        for col in range(min(COLS, len(v_lines) - 1)):
            x1 = max(0, v_lines[col])
            x2 = min(board_image.shape[1], v_lines[col + 1])
            cells.append(board_image[y1:y2, x1:x2])

    return cells


def split_board_with_positions(board_image: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = _detect_grid_lines(gray, board_image.shape[:2])

    cells = []
    for row in range(min(ROWS, len(h_lines) - 1)):
        y1 = max(0, h_lines[row])
        y2 = min(board_image.shape[0], h_lines[row + 1])
        for col in range(min(COLS, len(v_lines) - 1)):
            x1 = max(0, v_lines[col])
            x2 = min(board_image.shape[1], v_lines[col + 1])
            cells.append((row, col, board_image[y1:y2, x1:x2]))

    return cells


def _detect_grid_lines(gray: np.ndarray, shape: tuple) -> tuple[list[int], list[int]]:
    h, w = shape

    h_lines = _detect_by_sobel(gray, axis='h', expected=ROWS + 1)
    v_lines = _detect_by_sobel(gray, axis='v', expected=COLS + 1)

    if len(h_lines) < ROWS + 1 or len(v_lines) < COLS + 1:
        hh, vv = _detect_by_hough(gray, h, w)
        if len(hh) >= ROWS + 1:
            h_lines = hh
        else:
            h_lines = _fill_lines(h_lines, ROWS + 1, h)
        if len(vv) >= COLS + 1:
            v_lines = vv
        else:
            v_lines = _fill_lines(v_lines, COLS + 1, w)

    if len(h_lines) > ROWS + 1:
        h_lines = _trim_lines(h_lines, ROWS + 1)
    if len(v_lines) > COLS + 1:
        v_lines = _trim_lines(v_lines, COLS + 1)

    return h_lines, v_lines


def _detect_by_sobel(gray: np.ndarray, axis: str, expected: int) -> list[int]:
    if axis == 'h':
        edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        profile = edge.mean(axis=1)
    else:
        edge = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        profile = edge.mean(axis=0)

    profile = np.convolve(profile, np.ones(5) / 5, mode='same')

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