import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image, make_grid
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class CelebAHQDataset(ImageFolder):
    """
    Dataset class for CelebA-HQ.
    """
    def __init__(self, root_dir, mode='train', transform=None, null_context=False):
        """
        Args:
            root_dir: Path to root directory containing train/ and val/ folders
            mode: 'train' or 'val' to specify which split to use
            transform: Optional transform to be applied on images
            null_context: If True, return 0 as label (for unconditional generation)
        """
        mode_dir = Path(root_dir) / mode
        super().__init__(root=str(mode_dir), transform=transform)

        self.null_context = null_context
        self.n_classes = len(self.classes)

        print(f"Found {len(self.samples)} images in {mode_dir}")
        print(f"Classes: {self.classes}")
        print(f"Class to index mapping: {self.class_to_idx}")

        # Count samples per class
        for class_name, class_idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == class_idx)
            print(f"  - {class_name}: {count}")

    def __getitem__(self, idx):
        image, label_idx = super().__getitem__(idx)

        if self.null_context:
            # For unconditional generation
            label = torch.tensor(0).to(torch.int64)
        else:
            # Convert integer label to one-hot encoding
            # female (idx=0) -> [1, 0], male (idx=1) -> [0, 1]
            label = torch.zeros(self.n_classes)
            label[label_idx] = 1
            label = label.to(torch.int64)

        return (image, label)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # range [-1,1]
])
