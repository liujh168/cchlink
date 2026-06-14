import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
)

NUM_CLASSES = 15

CLASS_NAMES = [
    "红帅",
    "红仕",
    "红相",
    "红俥",
    "红马",
    "红炮",
    "红兵",
    "黑将",
    "黑士",
    "黑象",
    "黑车",
    "黑马",
    "黑炮",
    "黑卒",
    "空",
]

BACKBONES = {
    "mobilenet_v3_small": {
        "builder": mobilenet_v3_small,
        "weights": MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "classifier_attr": "classifier",
        "in_features_index": 3,
    },
    "mobilenet_v3_large": {
        "builder": mobilenet_v3_large,
        "weights": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "classifier_attr": "classifier",
        "in_features_index": 3,
    },
    "resnet18": {
        "builder": resnet18,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "classifier_attr": "fc",
    },
    "resnet34": {
        "builder": resnet34,
        "weights": ResNet34_Weights.IMAGENET1K_V1,
        "classifier_attr": "fc",
    },
    "efficientnet_b0": {
        "builder": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "classifier_attr": "classifier",
        "in_features_index": 1,
    },
}


def build_model(
    num_classes: int = NUM_CLASSES,
    backbone: str = "mobilenet_v3_small",
    pretrained: bool = True,
) -> nn.Module:
    """构建指定骨干网络，并将最终分类层替换为棋子类别数。"""
    if backbone not in BACKBONES:
        raise ValueError(f"未知骨干网络: {backbone}，可选值: {list(BACKBONES.keys())}")

    cfg = BACKBONES[backbone]
    weights = cfg["weights"] if pretrained else None
    model = cfg["builder"](weights=weights)

    classifier_attr = cfg["classifier_attr"]
    in_features_index = cfg.get("in_features_index")

    if in_features_index is not None:
        classifier = getattr(model, classifier_attr)
        in_features = classifier[in_features_index].in_features
        classifier[in_features_index] = nn.Linear(in_features, num_classes)
    else:
        classifier = getattr(model, classifier_attr)
        in_features = classifier.in_features
        setattr(model, classifier_attr, nn.Linear(in_features, num_classes))

    return model


def save_model(model: nn.Module, path: str, metadata: dict | None = None):
    """保存模型权重；提供元数据时同时记录骨干网络和预处理配置。"""
    if metadata is None:
        torch.save(model.state_dict(), path)
        return
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_model(
    path: str,
    num_classes: int = NUM_CLASSES,
    backbone: str = "mobilenet_v3_small",
    device: str = "cpu",
) -> nn.Module:
    """加载新旧两种 checkpoint，并优先采用文件内记录的模型配置。"""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_backbone = metadata.get("backbone", backbone)
    model = build_model(num_classes=num_classes, backbone=checkpoint_backbone, pretrained=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.metadata = metadata
    model.to(device)
    model.eval()
    return model
