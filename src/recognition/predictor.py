import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.recognition.model import load_model, CLASS_NAMES, NUM_CLASSES
from src.recognition.dataset import VAL_TRANSFORM

PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PiecePredictor:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = load_model(model_path, num_classes=NUM_CLASSES, device=device)

    def predict_cell(self, cell_image: np.ndarray) -> int:
        pil_image = Image.fromarray(cell_image)
        input_tensor = PREDICT_TRANSFORM(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()

    def predict_grid(self, cells: list[np.ndarray]) -> list[int]:
        predictions = []
        for cell in cells:
            pred = self.predict_cell(cell)
            predictions.append(pred)
        return predictions

    def predict_grid_with_positions(
        self, cells_with_pos: list[tuple[int, int, np.ndarray]]
    ) -> list[tuple[int, int, int, str]]:
        results = []
        for row, col, cell in cells_with_pos:
            pred = self.predict_cell(cell)
            class_name = CLASS_NAMES[pred]
            results.append((row, col, pred, class_name))
        return results
