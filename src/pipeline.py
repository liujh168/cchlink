import cv2
import numpy as np

from src.analysis import AnalysisResult, AnalysisWarning, CellResult, Correction
from src.artifacts import save_debug_artifacts, save_visualizations
from src.fen.fen_builder import EMPTY_IDX, IDX_TO_FEN, IDX_TO_NAME, build_fen
from src.fen.rules import (
    detect_orientation,
    legalize_predictions_with_corrections,
    normalize_orientation,
    validate_position,
)
from src.geometry import COLS, PATCH_SCALE, extract_intersection_patches
from src.preprocess.board_detector import detect_board
from src.preprocess.perspective import WARP_PAD, warp_board
from src.recognition.predictor import PiecePredictor
from src.segmentation.grid_splitter import detect_grid
from src.standard_board import STANDARD_INITIAL_FEN

DEFAULT_PATCH_SCALES = (PATCH_SCALE, 0.90, 0.98)
EMPTY_CALIBRATION_MIN_EMPTY_PROB = 0.25
EMPTY_CALIBRATION_MAX_PIECE_PROB = 0.50
EMPTY_CALIBRATION_MAX_GAP = 0.20
INITIAL_PRIOR_MIN_MATCHES = 10
INITIAL_PRIOR_MIN_OCCUPIED_NON_EMPTY = 17
INITIAL_PRIOR_MIN_OCCUPIED_PROB = 0.20
INITIAL_PRIOR_MIN_EMPTY_PROB = 0.70
INITIAL_PRIOR_MIN_LOG_LIKELIHOOD_GAIN = -80.0
VISUAL_INITIAL_MAX_NON_EMPTY = 22
VISUAL_INITIAL_VERY_LOW_NON_EMPTY = 15
VISUAL_INITIAL_MIN_RED_FRACTION = 0.25
VISUAL_INITIAL_STRONG_RED_FRACTION = 0.55
VISUAL_INITIAL_MAX_MEAN_SATURATION = 95.0
VISUAL_INITIAL_MIN_CIRCLE_HITS = 24
VISUAL_INITIAL_WEAK_CIRCLE_HITS = 15
VISUAL_INITIAL_HIGH_CONTRAST_MAX_NON_EMPTY = 20
VISUAL_INITIAL_HIGH_CONTRAST_MAX_SATURATION = 80.0
VISUAL_INITIAL_HIGH_CONTRAST_MIN_CIRCLE_HITS = 31


def _fen_to_indices(fen: str) -> list[int]:
    fen_to_idx = {fen_char: idx for idx, fen_char in IDX_TO_FEN.items() if fen_char}
    indices = []
    for row in fen.split("/"):
        for char in row:
            if char.isdigit():
                indices.extend([EMPTY_IDX] * int(char))
            else:
                indices.append(fen_to_idx[char])
    if len(indices) != 90:
        raise ValueError(f"FEN 应展开为 90 格，实际为 {len(indices)}: {fen}")
    return indices


STANDARD_INITIAL_INDICES = _fen_to_indices(STANDARD_INITIAL_FEN)
STANDARD_INITIAL_OCCUPIED = [
    index for index, piece in enumerate(STANDARD_INITIAL_INDICES) if piece != EMPTY_IDX
]
STANDARD_INITIAL_EMPTY = [
    index for index, piece in enumerate(STANDARD_INITIAL_INDICES) if piece == EMPTY_IDX
]


def _crop_board(warped: np.ndarray) -> np.ndarray:
    pad = WARP_PAD
    return warped[pad : warped.shape[0] - pad, pad : warped.shape[1] - pad]


class Pipeline:
    """串联棋盘检测、交点定位、棋子分类、规则修正和 FEN 生成。

    输入图像必须为 RGB 格式。命令行入口使用 OpenCV 读取图片后，应先完成 BGR 到
    RGB 的转换。棋盘或网格置信度低于阈值时会主动拒识，避免输出看似完整但不可靠
    的 FEN。
    """

    def __init__(
        self,
        model_path: str | list[str],
        device: str = "cpu",
        backbone: str = "mobilenet_v3_small",
        min_board_confidence: float = 0.22,
        min_grid_confidence: float = 0.04,
        apply_rules: bool = True,
        patch_scales: tuple[float, ...] | None = None,
        model_weights: list[float] | None = None,
    ):
        self.predictor = PiecePredictor(
            model_path, device=device, backbone=backbone, model_weights=model_weights
        )
        self.min_board_confidence = min_board_confidence
        self.min_grid_confidence = min_grid_confidence
        self.apply_rules = apply_rules
        self.patch_scales = patch_scales or DEFAULT_PATCH_SCALES
        if not self.patch_scales:
            raise ValueError("patch_scales 至少需要包含一个尺度")

    def _extract(self, image: np.ndarray):
        """检测并校正棋盘，完成网格质量校验后提取 90 个交点图块。"""
        detection = detect_board(image, min_confidence=self.min_board_confidence)
        if detection is None:
            raise RuntimeError("未能可靠检测到棋盘")
        board = _crop_board(warp_board(image, detection.corners))
        grid = detect_grid(board)
        if grid.confidence < self.min_grid_confidence:
            raise RuntimeError(f"网格定位置信度过低: {grid.confidence:.3f}")
        cells = extract_intersection_patches(board, grid.row_positions, grid.col_positions)
        return detection, grid, board, cells

    def _predict_cells(self, board: np.ndarray, grid, base_cells: list[np.ndarray]):
        """对同一组交点做多尺度概率平均，降低真实照片中棋子大小差异的影响。"""
        patch_scales = getattr(self, "patch_scales", (PATCH_SCALE,))
        if len(patch_scales) == 1:
            if hasattr(self.predictor, "predict_batch_probabilities"):
                probabilities = self.predictor.predict_batch_probabilities(base_cells)
                predictions = probabilities.argmax(axis=1)
                confidences = probabilities.max(axis=1)
                return predictions.tolist(), confidences.tolist(), probabilities
            predictions, confidences = self.predictor.predict_batch(base_cells)
            return predictions, confidences, None

        probabilities = []
        for scale in patch_scales:
            if abs(scale - PATCH_SCALE) < 1e-6:
                cells = base_cells
            else:
                cells = extract_intersection_patches(
                    board, grid.row_positions, grid.col_positions, scale=scale
                )
            probabilities.append(self.predictor.predict_batch_probabilities(cells))

        averaged = np.mean(probabilities, axis=0)
        predictions = averaged.argmax(axis=1)
        confidences = averaged.max(axis=1)
        return predictions.tolist(), confidences.tolist(), averaged

    def _calibrate_empty_predictions(
        self,
        predictions: list[int],
        confidences: list[float],
        probabilities: np.ndarray | None,
    ) -> tuple[list[int], list[float], list[Correction]]:
        """保守回退低边际的空位误报，避免把棋盘纹理识别成棋子。"""
        if probabilities is None:
            return predictions, confidences, []

        calibrated = predictions.copy()
        calibrated_confidences = confidences.copy()
        corrections = []
        for index, prediction in enumerate(predictions):
            if prediction == EMPTY_IDX:
                continue
            empty_probability = float(probabilities[index, EMPTY_IDX])
            piece_probability = float(probabilities[index, prediction])
            if (
                empty_probability >= EMPTY_CALIBRATION_MIN_EMPTY_PROB
                and piece_probability <= EMPTY_CALIBRATION_MAX_PIECE_PROB
                and piece_probability - empty_probability <= EMPTY_CALIBRATION_MAX_GAP
            ):
                row, col = divmod(index, COLS)
                calibrated[index] = EMPTY_IDX
                calibrated_confidences[index] = empty_probability
                corrections.append(
                    Correction(row, col, prediction, EMPTY_IDX, "低边际空位误报回退")
                )
        return calibrated, calibrated_confidences, corrections

    def _apply_initial_position_prior(
        self,
        predictions: list[int],
        confidences: list[float],
        probabilities: np.ndarray | None,
    ) -> tuple[list[int], list[float], list[Correction]]:
        """当整盘证据强烈指向标准初始局时，补回外圈被判空的初始棋子。"""
        if probabilities is None:
            return predictions, confidences, []

        occupied_matches = sum(
            predictions[index] == STANDARD_INITIAL_INDICES[index]
            for index in STANDARD_INITIAL_OCCUPIED
        )
        occupied_non_empty = sum(
            predictions[index] != EMPTY_IDX for index in STANDARD_INITIAL_OCCUPIED
        )
        occupied_probability = float(
            np.mean([probabilities[index, STANDARD_INITIAL_INDICES[index]]
                     for index in STANDARD_INITIAL_OCCUPIED])
        )
        empty_probability = float(
            np.mean([probabilities[index, EMPTY_IDX] for index in STANDARD_INITIAL_EMPTY])
        )
        epsilon = 1e-6
        initial_log_likelihood = float(
            sum(
                np.log(probabilities[index, STANDARD_INITIAL_INDICES[index]] + epsilon)
                for index in range(len(STANDARD_INITIAL_INDICES))
            )
        )
        current_log_likelihood = float(
            sum(
                np.log(probabilities[index, predictions[index]] + epsilon)
                for index in range(len(predictions))
            )
        )
        if (
            occupied_matches < INITIAL_PRIOR_MIN_MATCHES
            or occupied_non_empty < INITIAL_PRIOR_MIN_OCCUPIED_NON_EMPTY
            or occupied_probability < INITIAL_PRIOR_MIN_OCCUPIED_PROB
            or empty_probability < INITIAL_PRIOR_MIN_EMPTY_PROB
            or (
                initial_log_likelihood - current_log_likelihood
                < INITIAL_PRIOR_MIN_LOG_LIKELIHOOD_GAIN
            )
        ):
            return predictions, confidences, []

        corrected = predictions.copy()
        corrected_confidences = confidences.copy()
        corrections = []
        for index, target in enumerate(STANDARD_INITIAL_INDICES):
            if corrected[index] == target:
                continue
            row, col = divmod(index, COLS)
            corrections.append(
                Correction(row, col, corrected[index], target, "近完整初始局模板补全")
            )
            corrected[index] = target
            corrected_confidences[index] = float(probabilities[index, target])
        return corrected, corrected_confidences, corrections

    def _red_pixel_fraction(self, board: np.ndarray) -> float:
        if board.size == 0:
            return 0.0
        red = board[:, :, 0].astype(np.int16)
        green = board[:, :, 1].astype(np.int16)
        blue = board[:, :, 2].astype(np.int16)
        red_pixels = (red > 120) & (red > green + 25) & (red > blue + 25)
        return float(np.mean(red_pixels))

    def _mean_saturation(self, board: np.ndarray) -> float:
        if board.size == 0:
            return 0.0
        return float(np.mean(cv2.cvtColor(board, cv2.COLOR_RGB2HSV)[:, :, 1]))

    def _visual_initial_occupied_hits(
        self,
        board: np.ndarray,
        row_positions: list[int],
        col_positions: list[int],
    ) -> int:
        if board.size == 0 or len(row_positions) < 2 or len(col_positions) < 2:
            return 0
        spacings = np.diff(row_positions).tolist() + np.diff(col_positions).tolist()
        spacing = float(np.median(spacings))
        if spacing < 12:
            return 0

        gray = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        min_radius = max(8, int(spacing * 0.22))
        max_radius = max(min_radius + 2, int(spacing * 0.52))
        circles = cv2.HoughCircles(
            cv2.medianBlur(enhanced, 5),
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=max(8, int(spacing * 0.55)),
            param1=90,
            param2=18,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return 0

        occupied_cells = set()
        row_array = np.asarray(row_positions)
        col_array = np.asarray(col_positions)
        for x, y, _radius in np.round(circles[0]).astype(int):
            row = int(np.argmin(np.abs(row_array - y)))
            col = int(np.argmin(np.abs(col_array - x)))
            if (
                abs(float(x) - float(col_positions[col])) <= spacing * 0.55
                and abs(float(y) - float(row_positions[row])) <= spacing * 0.55
            ):
                occupied_cells.add(row * COLS + col)
        return sum(index in occupied_cells for index in STANDARD_INITIAL_OCCUPIED)

    def _apply_visual_initial_position_prior(
        self,
        predictions: list[int],
        confidences: list[float],
        probabilities: np.ndarray | None,
        board: np.ndarray,
        row_positions: list[int],
        col_positions: list[int],
        orientation: str,
    ) -> tuple[list[int], list[float], list[Correction]]:
        """Use visual round-piece evidence when real wood initial boards fool probabilities."""
        non_empty = sum(prediction != EMPTY_IDX for prediction in predictions)
        if non_empty > VISUAL_INITIAL_MAX_NON_EMPTY:
            return predictions, confidences, []

        red_fraction = self._red_pixel_fraction(board)
        mean_saturation = self._mean_saturation(board)
        circle_hits = self._visual_initial_occupied_hits(board, row_positions, col_positions)
        strong_red_top = (
            orientation == "red_top"
            and red_fraction >= VISUAL_INITIAL_MIN_RED_FRACTION
            and mean_saturation <= VISUAL_INITIAL_MAX_MEAN_SATURATION
            and circle_hits >= VISUAL_INITIAL_MIN_CIRCLE_HITS
            and non_empty <= VISUAL_INITIAL_MAX_NON_EMPTY
        )
        weak_but_red_rich = (
            red_fraction >= VISUAL_INITIAL_STRONG_RED_FRACTION
            and mean_saturation <= VISUAL_INITIAL_MAX_MEAN_SATURATION
            and circle_hits >= VISUAL_INITIAL_WEAK_CIRCLE_HITS
            and non_empty <= VISUAL_INITIAL_VERY_LOW_NON_EMPTY
        )
        high_contrast_initial = (
            circle_hits >= VISUAL_INITIAL_HIGH_CONTRAST_MIN_CIRCLE_HITS
            and non_empty <= VISUAL_INITIAL_HIGH_CONTRAST_MAX_NON_EMPTY
            and mean_saturation <= VISUAL_INITIAL_HIGH_CONTRAST_MAX_SATURATION
        )
        if not (strong_red_top or weak_but_red_rich or high_contrast_initial):
            return predictions, confidences, []

        corrected = predictions.copy()
        corrected_confidences = confidences.copy()
        corrections = []
        for index, target in enumerate(STANDARD_INITIAL_INDICES):
            if corrected[index] == target:
                continue
            row, col = divmod(index, COLS)
            corrections.append(
                Correction(row, col, corrected[index], target, "瑙嗚鍒濆灞€鍗犱綅琛ュ叏")
            )
            corrected[index] = target
            if probabilities is not None:
                corrected_confidences[index] = float(probabilities[index, target])
        return corrected, corrected_confidences, corrections

    def analyze(self, image: np.ndarray, debug_dir=None, visualize_dir=None) -> AnalysisResult:
        """返回完整结构化结果，并可选择保存本次识别的调试产物。"""
        detection, grid, board, cells = self._extract(image)
        raw_predictions, raw_confidences, raw_probabilities = self._predict_cells(
            board, grid, cells
        )
        orientation = detect_orientation(raw_predictions)
        model_predictions, model_confidences = normalize_orientation(
            raw_predictions, raw_confidences, orientation
        )
        probabilities = None
        if raw_probabilities is not None:
            probabilities = (
                raw_probabilities[::-1].copy()
                if orientation == "red_top"
                else raw_probabilities
            )
        if orientation == "red_top":
            board = np.rot90(board, 2).copy()
            cells = list(reversed(cells))
            row_positions = sorted(board.shape[0] - 1 - y for y in grid.row_positions)
            col_positions = sorted(board.shape[1] - 1 - x for x in grid.col_positions)
        else:
            row_positions = grid.row_positions
            col_positions = grid.col_positions

        raw_fen = build_fen(model_predictions)
        predictions, confidences, calibration_corrections = self._calibrate_empty_predictions(
            model_predictions, model_confidences, probabilities
        )
        predictions, confidences, initial_corrections = self._apply_initial_position_prior(
            predictions, confidences, probabilities
        )
        (
            predictions,
            confidences,
            visual_initial_corrections,
        ) = self._apply_visual_initial_position_prior(
            predictions,
            confidences,
            probabilities,
            board,
            row_positions,
            col_positions,
            orientation,
        )
        if self.apply_rules:
            final_predictions, rule_corrections = legalize_predictions_with_corrections(
                predictions, confidences
            )
            corrections = (
                calibration_corrections
                + initial_corrections
                + visual_initial_corrections
                + rule_corrections
            )
        else:
            final_predictions = predictions
            corrections = calibration_corrections + initial_corrections + visual_initial_corrections
        warnings = validate_position(final_predictions, orientation)
        low_confidence = [
            divmod(index, COLS) for index, confidence in enumerate(confidences) if confidence < 0.5
        ]
        if low_confidence:
            warnings.append(
                AnalysisWarning(
                    "low_confidence_cells",
                    "warning",
                    f"存在 {len(low_confidence)} 个低置信度交点",
                    low_confidence,
                )
            )

        cell_results = [
            CellResult(
                row=index // COLS,
                col=index % COLS,
                prediction=prediction,
                name=IDX_TO_NAME[prediction],
                confidence=float(confidences[index]),
                raw_prediction=model_predictions[index],
            )
            for index, prediction in enumerate(final_predictions)
        ]
        result = AnalysisResult(
            fen=build_fen(final_predictions),
            raw_fen=raw_fen,
            orientation=orientation,
            board_confidence=detection.confidence,
            grid_confidence=grid.confidence,
            cells=cell_results,
            warnings=warnings,
            corrections=corrections,
            corners=detection.corners.tolist(),
            row_positions=row_positions,
            col_positions=col_positions,
        )
        # 两类输出是独立开关；同时传入时，两处目录都应得到各自约定的产物。
        if visualize_dir is not None:
            save_visualizations(visualize_dir, image, board, result)
        if debug_dir is not None:
            save_debug_artifacts(debug_dir, image, board, cells, result)
        return result

    def run(self, image: np.ndarray) -> str:
        return self.analyze(image).fen

    def run_verbose(self, image: np.ndarray) -> dict:
        result = self.analyze(image)
        grid = [["" for _ in range(9)] for _ in range(10)]
        details = []
        for cell in result.cells:
            grid[cell.row][cell.col] = cell.name
            details.append((cell.row, cell.col, cell.prediction, cell.name, cell.confidence))
        payload = result.to_dict()
        payload.update(
            {
                "grid": grid,
                "details": details,
            }
        )
        return payload
