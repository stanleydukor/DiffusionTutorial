"""Dataset utilities for Part 2 — Stable Diffusion LoRA training."""

from pathlib import Path

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from torchvision import transforms


def load_wikiart(save_path="/mnt/f/Datasets/WikiArt"):
    """Load WikiArt from disk, or download and cache on first run."""
    path = Path(save_path)
    if path.exists() and any(path.iterdir()):
        print(f"Loading dataset from {path} …")
        dataset = load_from_disk(str(path))
    else:
        print("Downloading huggan/wikiart (~24 GB) …")
        dataset = load_dataset("huggan/wikiart")
        path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(path))
        print(f"Saved to {path}")

    train_split = dataset["train"]
    print(f"WikiArt loaded: {len(train_split):,} images")
    return train_split


class WikiArtStyleDataset(Dataset):
    """Single-style slice of WikiArt with (image_tensor, caption) pairs."""

    def __init__(self, hf_dataset, style_id, style_name, image_size=1024, max_samples=None):
        self.style_name = style_name

        self.data = hf_dataset.filter(
            lambda x: x["style"] == style_id,
            desc=f"Filtering {style_name}",
        )
        if max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(self.data))))
        print(f"  {style_name}: {len(self.data):,} images")

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = self.transform(sample["image"].convert("RGB"))
        caption = f"a painting in the style of {self.style_name.replace('_', ' ').lower()}"
        return {"image": image, "caption": caption}
