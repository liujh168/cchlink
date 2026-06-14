import argparse
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_SITE = os.path.join(PROJECT_ROOT, ".venv", "Lib", "site-packages")
if os.path.exists(VENV_SITE) and VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.recognition.model import build_model, save_model, NUM_CLASSES, CLASS_NAMES
from src.recognition.dataset import PieceDataset
from src.recognition.split import group_split_indices


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class _TransformSubset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = PieceDataset._load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def compute_class_weights(dataset, class_count=15):
    label_counts = np.zeros(class_count, dtype=np.float32)
    for _, label in dataset.samples:
        label_counts[label] += 1
    label_counts = np.maximum(label_counts, 1)
    weights = 1.0 / np.sqrt(label_counts)
    weights = weights / weights.sum() * class_count
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for cls in range(NUM_CLASSES):
            mask = labels == cls
            class_total[cls] += mask.sum().item()
            class_correct[cls] += (predicted[mask] == cls).sum().item()

    return total_loss / total, correct / total, class_correct, class_total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for cls in range(NUM_CLASSES):
            mask = labels == cls
            class_total[cls] += mask.sum().item()
            class_correct[cls] += (predicted[mask] == cls).sum().item()

        for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            confusion[t][p] += 1

    return total_loss / total, correct / total, class_correct, class_total, confusion


def print_class_accuracies(class_correct, class_total, prefix="  "):
    for cls in range(NUM_CLASSES):
        if class_total[cls] > 0:
            acc = class_correct[cls] / class_total[cls]
            print(
                f"{prefix}{CLASS_NAMES[cls]:6s}: {class_correct[cls]:.0f}/{class_total[cls]:.0f} ({acc:.1%})"
            )


def print_confusion_summary(confusion):
    print("  主要误判模式 (>=10例):")
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            if t != p and confusion[t][p] >= 10:
                print(f"    {CLASS_NAMES[t]} -> {CLASS_NAMES[p]}: {confusion[t][p]} 例")


def main():
    parser = argparse.ArgumentParser(description="训练棋子识别 CNN 模型 (v2)")
    parser.add_argument("--data", "-d", required=True, help="训练数据根目录")
    parser.add_argument("--epochs", "-e", type=int, default=40, help="训练轮数")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--output", "-o", required=True, help="模型输出路径 (*.pth)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="训练设备",
    )
    parser.add_argument("--val_split", type=float, default=0.2, help="验证集比例")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=[
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "resnet18",
            "resnet34",
            "efficientnet_b0",
        ],
        help="模型骨干网络",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="焦点损失的 gamma 参数")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    from src.recognition.dataset import TRAIN_TRANSFORM, VAL_TRANSFORM

    full_dataset = PieceDataset(args.data, transform=None)
    train_indices, val_indices = group_split_indices(full_dataset.groups, args.val_split)
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    train_size = len(train_samples)
    val_size = len(val_samples)

    train_dataset = _TransformSubset(train_samples, TRAIN_TRANSFORM)
    val_dataset = _TransformSubset(val_samples, VAL_TRANSFORM)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(full_dataset)
    print(f"类别权重: {class_weights.tolist()}")
    class_weights = class_weights.to(device)

    print(f"训练集: {train_size} 样本, 验证集: {val_size} 样本")

    model = build_model(num_classes=NUM_CLASSES, backbone=args.backbone, pretrained=True)
    model = model.to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_cls_corr, train_cls_tot = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_cls_corr, val_cls_tot, confusion = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model,
                args.output,
                metadata={
                    "backbone": args.backbone,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "input_size": [48, 48],
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                    "best_val_accuracy": best_val_acc,
                },
            )
            print(f"  -> 保存最佳模型到 {args.output}")

        if epoch % 5 == 0 or epoch == 1:
            print_class_accuracies(val_cls_corr, val_cls_tot)

    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")
    print("\n最终验证集各类准确率:")
    print_class_accuracies(val_cls_corr, val_cls_tot)
    print_confusion_summary(confusion)


if __name__ == "__main__":
    main()
