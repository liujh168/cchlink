import csv
import hashlib
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_TO_IDX = {
    "红帅": 0,
    "红仕": 1,
    "红相": 2,
    "红俥": 3,
    "红马": 4,
    "红炮": 5,
    "红兵": 6,
    "黑将": 7,
    "黑士": 8,
    "黑象": 9,
    "黑车": 10,
    "黑马": 11,
    "黑炮": 12,
    "黑卒": 13,
    "空": 14,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.RandomAffine(degrees=25, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=8),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.6),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PieceDataset(Dataset):
    """加载棋子分类数据，并保留用于防止数据泄漏的棋盘分组标识。

    优先读取带有 path、label、group 字段的 manifest.csv。旧式按类别目录组织的
    数据仍可使用，但每张图片会被视为独立分组，只适合作为兼容模式。
    """

    def __init__(self, root_dir: str, transform=None):
        self.samples = []
        self.groups = []
        self.provenance = []
        self.transform = transform
        self.manifest_path = None
        self.manifest_sha256 = None

        manifest_path = os.path.join(root_dir, "manifest.csv")
        if os.path.exists(manifest_path):
            self.manifest_path = manifest_path
            with open(manifest_path, "rb") as handle:
                self.manifest_sha256 = hashlib.sha256(handle.read()).hexdigest()
            self._load_manifest(root_dir, manifest_path)
            return

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
                    self.groups.append(os.path.join(class_name, fname))
                    self.provenance.append({})

    def _load_manifest(self, root_dir: str, manifest_path: str):
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                class_name = row["label"]
                if class_name not in CLASS_TO_IDX:
                    continue
                path = row["path"]
                if not os.path.isabs(path):
                    path = os.path.join(root_dir, path)
                self.samples.append((path, CLASS_TO_IDX[class_name]))
                self.groups.append(row["group"])
                self.provenance.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._load_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def _load_image(img_path: str) -> Image.Image:
        return Image.open(img_path).convert("RGB")
