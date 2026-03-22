"""
Dataset Module

This module provides:
- load_wikiart(): Load or download the WikiArt dataset (huggan/wikiart on HuggingFace Hub)
- WikiArtStyleDataset: PyTorch Dataset for a single artistic style with image and caption pairs
"""

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset, load_from_disk


def load_wikiart(save_path="/mnt/f/Datasets/WikiArt"):
    """
    Load WikiArt from local disk, or download and cache it on first run

    Args:
        save_path: directory to save / load the dataset
    Returns:
        HuggingFace Dataset (train split) with image, artist, style, genre columns
    """
    path = Path(save_path)

    if path.exists() and any(path.iterdir()):
        print(f"Dataset found at {path}. Loading from disk …")
        dataset = load_from_disk(str(path))
    else:
        print("Downloading huggan/wikiart from Hugging Face Hub …")
        print("(First download is ~24 GB — this will take a while)")
        dataset = load_dataset("huggan/wikiart")
        print(f"Saving dataset to {path} …")
        path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(path))
        print("✓ Dataset saved — subsequent runs will load from disk instantly")

    train_split = dataset["train"]
    print(f"WikiArt loaded: {len(train_split):,} images")
    print(f"Features      : {list(train_split.features.keys())}")
    return train_split


class WikiArtStyleDataset(Dataset):
    """
    PyTorch Dataset for a single artistic style from WikiArt

    Filters the WikiArt dataset to the target style and returns
    (image_tensor, caption) pairs for LoRA fine-tuning.

    Args:
        hf_dataset: HuggingFace Dataset (train split of huggan/wikiart)
        style_id: integer class index for the target style
        style_name: human-readable style name (used in captions)
        image_size: target image resolution (default 1024 for SDXL)
        max_samples: maximum number of images to use (None = use all)
    """

    def __init__(self, hf_dataset, style_id, style_name, image_size=1024, max_samples=None):
        self.style_name = style_name
        self.image_size = image_size

        # Filter the full dataset to only the target style
        self.data = hf_dataset.filter(
            lambda x: x["style"] == style_id,
            desc=f"Filtering for {style_name}",
        )

        if max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(self.data))))

        print(f"  {style_name}: {len(self.data):,} images")

        # Resize → centre-crop → normalise to [-1, 1] (VAE input convention)
        self.transform = transforms.Compose([
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image  = sample["image"].convert("RGB")
        image  = self.transform(image)

        # Caption tells the model which style this LoRA adapter should learn
        caption = (
            f"a painting in the style of "
            f"{self.style_name.replace('_', ' ').lower()}"
        )

        return {"image": image, "caption": caption}
