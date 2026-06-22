"""
MVTec AD Dataset Loader
-----------------------
Loads train (normal only) + test (normal + defective) splits.
Returns tensors ready for feature extraction.

Why these transforms?
  - Resize(256) → CenterCrop(224): standard ImageNet preprocessing.
    WideResNet was pretrained on ImageNet at 224×224 — mismatched
    input size kills feature quality.
  - Normalize(mean, std): same ImageNet stats the backbone was trained
    with. Skipping this degrades AUROC by ~5–10 points.
  - No heavy augmentation on train: PatchCore memorises normal patches,
    not a classifier — augmentation would corrupt the memory bank.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── constants ──────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TEST_TRANSFORMS = TRAIN_TRANSFORMS   # identical — no augmentation at test time


# ── dataset ────────────────────────────────────────────────────────────────
class MVTecDataset(Dataset):
    """
    Expects MVTec folder structure:
        <root>/<category>/train/good/*.png
        <root>/<category>/test/good/*.png
        <root>/<category>/test/<defect_type>/*.png
        <root>/<category>/ground_truth/<defect_type>/*.png

    Args:
        root      : path to mvtec_anomaly_detection/
        category  : one of the 15 categories e.g. 'bottle', 'carpet'
        split     : 'train' or 'test'
        transform : torchvision transform pipeline
    """

    def __init__(self, root: str, category: str, split: str = "train",
                 transform=None):
        self.root      = Path(root) / category
        self.split     = split
        self.transform = transform or (TRAIN_TRANSFORMS if split == "train"
                                       else TEST_TRANSFORMS)
        self.samples   = self._load_samples()

    def _load_samples(self):
        samples = []
        split_dir = self.root / self.split

        for defect_type in sorted(split_dir.iterdir()):
            label = 0 if defect_type.name == "good" else 1
            for img_path in sorted(defect_type.glob("*.png")):
                # locate ground-truth mask (only exists for defective test images)
                mask_path = None
                if label == 1:
                    mask_path = (self.root / "ground_truth"
                                 / defect_type.name
                                 / img_path.name.replace(".png", "_mask.png"))
                    if not mask_path.exists():
                        mask_path = None
                samples.append((img_path, label, mask_path))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # mask: binary tensor H×W  (1 = anomaly pixel, 0 = normal)
        mask = torch.zeros(1, 224, 224)
        if mask_path and mask_path.exists():
            raw = Image.open(mask_path).convert("L")
            raw = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])(raw)
            mask = (raw > 0.5).float()

        return {
            "image"    : image,
            "label"    : torch.tensor(label, dtype=torch.long),
            "mask"     : mask,
            "path"     : str(img_path),
        }


# ── convenience loaders ────────────────────────────────────────────────────
def get_loaders(root: str, category: str, batch_size: int = 32,
                num_workers: int = 4):
    """Returns (train_loader, test_loader)."""
    train_ds = MVTecDataset(root, category, split="train")
    test_ds  = MVTecDataset(root, category, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader
