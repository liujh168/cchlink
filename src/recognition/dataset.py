import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


CLASS_TO_IDX = {
    "红帅": 0, "红仕": 1, "红相": 2, "红俥": 3, "红马": 4, "红炮": 5, "红兵": 6,
    "黑将": 7, "黑士": 8, "黑象": 9, "黑车": 10, "黑马": 11, "黑炮": 12, "黑卒": 13,
    "空": 14,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PieceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.transform = transform

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
