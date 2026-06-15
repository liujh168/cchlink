import numpy as np

from src.analysis import AnalysisResult, AnalysisWarning, CellResult
from src.artifacts import save_debug_artifacts, save_visualizations
from src.fen.fen_builder import IDX_TO_NAME, build_fen
from src.fen.rules import (
    detect_orientation,
    legalize_predictions_with_corrections,
    normalize_orientation,
    validate_position,
)
from src.geometry import COLS, extract_intersection_patches
from src.preprocess.board_detector import detect_board
from src.preprocess.perspective import WARP_PAD, warp_board
from src.recognition.predictor import PiecePredictor
from src.segmentation.grid_splitter import detect_grid


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
        model_path: str,
        device: str = "cpu",
        backbone: str = "mobilenet_v3_small",
        min_board_confidence: float = 0.22,
        min_grid_confidence: float = 0.05,
        apply_rules: bool = True,
    ):
        self.predictor = PiecePredictor(model_path, device=device, backbone=backbone)
        self.min_board_confidence = min_board_confidence
        self.min_grid_confidence = min_grid_confidence
        self.apply_rules = apply_rules

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

    def analyze(self, image: np.ndarray, debug_dir=None, visualize_dir=None) -> AnalysisResult:
        """返回完整结构化结果，并可选择保存本次识别的调试产物。"""
        detection, grid, board, cells = self._extract(image)
        raw_predictions, raw_confidences = self.predictor.predict_batch(cells)
        orientation = detect_orientation(raw_predictions)
        predictions, confidences = normalize_orientation(
            raw_predictions, raw_confidences, orientation
        )
        if orientation == "red_top":
            board = np.rot90(board, 2).copy()
            cells = list(reversed(cells))
            row_positions = sorted(board.shape[0] - 1 - y for y in grid.row_positions)
            col_positions = sorted(board.shape[1] - 1 - x for x in grid.col_positions)
        else:
            row_positions = grid.row_positions
            col_positions = grid.col_positions

        raw_fen = build_fen(predictions)
        if self.apply_rules:
            final_predictions, corrections = legalize_predictions_with_corrections(
                predictions, confidences
            )
        else:
            final_predictions, corrections = predictions, []
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
                raw_prediction=predictions[index],
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
