import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


NUM_CLASSES = 15

CLASS_NAMES = [
    "红帅", "红仕", "红相", "红俥", "红马", "红炮", "红兵",
    "黑将", "黑士", "黑象", "黑车", "黑马", "黑炮", "黑卒",
    "空",
]


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
    else:
        model = mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, num_classes: int = NUM_CLASSES, device: str = "cpu") -> nn.Module:
    model = build_model(num_classes=num_classes)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
