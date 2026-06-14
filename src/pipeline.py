import numpy as np

from src.fen.fen_builder import IDX_TO_NAME, build_fen
from src.fen.rules import legalize_predictions
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
        return detection, grid, cells

    def run(self, image: np.ndarray) -> str:
        _, _, cells = self._extract(image)
        predictions, confidences = self.predictor.predict_batch(cells)
        if self.apply_rules:
            predictions = legalize_predictions(predictions, confidences)
        fen = build_fen(predictions)
        return fen

    def run_verbose(self, image: np.ndarray) -> dict:
        detection, grid_detection, cells = self._extract(image)
        cells_with_pos = [(index // COLS, index % COLS, cell) for index, cell in enumerate(cells)]
        predictions, confidences = self.predictor.predict_batch(cells)
        if self.apply_rules:
            predictions = legalize_predictions(predictions, confidences)
        results = [
            (row, col, pred, IDX_TO_NAME[pred], confidence)
            for (row, col, _), pred, confidence in zip(cells_with_pos, predictions, confidences)
        ]
        fen = build_fen(predictions)

        grid = [["" for _ in range(9)] for _ in range(10)]
        for row, col, _, name, _ in results:
            grid[row][col] = name

        return {
            "fen": fen,
            "grid": grid,
            "details": results,
            "board_confidence": detection.confidence,
            "grid_confidence": grid_detection.confidence,
            "corners": detection.corners.tolist(),
        }
