import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.recognition.model import CLASS_NAMES, NUM_CLASSES, load_model

PREDICT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PiecePredictor:
    """棋子分类推理封装，支持单格兼容接口和整盘批量推理。"""

    def __init__(self, model_path: str, device: str = "cpu", backbone: str = "mobilenet_v3_small"):
        self.device = device
        self.model = load_model(
            model_path, num_classes=NUM_CLASSES, backbone=backbone, device=device
        )

    def predict_cell(self, cell_image: np.ndarray) -> int:
        predictions, _ = self.predict_batch([cell_image])
        return predictions[0]

    def predict_batch(self, cells: list[np.ndarray]) -> tuple[list[int], list[float]]:
        """一次前向传播完成多个格子的分类，并返回类别及其最高概率。

        完整棋盘包含 90 个交点，将它们堆叠成批次可以显著减少逐格调用模型产生的
        Python 调度和算子启动开销。
        """
        probabilities = self.predict_batch_probabilities(cells)
        if probabilities.size == 0:
            return [], []
        predicted = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        return predicted.tolist(), confidence.tolist()

    def predict_batch_probabilities(self, cells: list[np.ndarray]) -> np.ndarray:
        """返回每个图块的完整类别概率，用于多尺度等融合策略。"""
        if not cells:
            return np.empty((0, NUM_CLASSES), dtype=np.float32)
        input_tensor = torch.stack([PREDICT_TRANSFORM(Image.fromarray(cell)) for cell in cells]).to(
            self.device
        )

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)

        return probabilities.cpu().numpy()

    def predict_grid(self, cells: list[np.ndarray]) -> list[int]:
        predictions, _ = self.predict_batch(cells)
        return predictions

    def predict_grid_with_positions(
        self, cells_with_pos: list[tuple[int, int, np.ndarray]]
    ) -> list[tuple[int, int, int, str, float]]:
        predictions, confidences = self.predict_batch([cell for _, _, cell in cells_with_pos])
        results = []
        for (row, col, _), pred, confidence in zip(cells_with_pos, predictions, confidences):
            class_name = CLASS_NAMES[pred]
            results.append((row, col, pred, class_name, confidence))
        return results
