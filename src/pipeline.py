import numpy as np

from src.preprocess.board_detector import detect_board_corners
from src.preprocess.perspective import warp_board
from src.segmentation.grid_splitter import split_board, split_board_with_positions
from src.recognition.predictor import PiecePredictor
from src.fen.fen_builder import build_fen, IDX_TO_NAME


class Pipeline:
    """Pipeline expects image input in RGB format.

    CLI scripts should convert images to RGB before calling run() or run_verbose().
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.predictor = PiecePredictor(model_path, device=device)

    def run(self, image: np.ndarray) -> str:
        corners = detect_board_corners(image)
        if corners is None:
            raise RuntimeError("未能检测到棋盘，请确保图片中包含完整的中国象棋棋盘")

        board = warp_board(image, corners)
        cells = split_board(board)
        predictions = self.predictor.predict_grid(cells)
        fen = build_fen(predictions)
        return fen

    def run_verbose(self, image: np.ndarray) -> dict:
        corners = detect_board_corners(image)
        if corners is None:
            raise RuntimeError("未能检测到棋盘，请确保图片中包含完整的中国象棋棋盘")

        board = warp_board(image, corners)
        cells_with_pos = split_board_with_positions(board)
        results = self.predictor.predict_grid_with_positions(cells_with_pos)

        predictions = [pred for _, _, pred, _ in results]
        fen = build_fen(predictions)

        grid = [["" for _ in range(9)] for _ in range(10)]
        for row, col, _, name in results:
            grid[row][col] = name

        return {"fen": fen, "grid": grid, "details": results}
