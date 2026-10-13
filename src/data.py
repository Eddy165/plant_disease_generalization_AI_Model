from __future__ import annotations
from collections import Counter
from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transforms(image_size=224, train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class HFDatasetAdapter(Dataset):
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])



def load_plantvillage(dataset_script="./plant_village.py"):
    """Load the official PlantVillage color dataset from a local script."""
    from pathlib import Path
    from datasets import load_dataset

    script = Path(dataset_script)

    if not script.exists():
        raise FileNotFoundError(
            f"Dataset script not found: {script.resolve()}. "
            "Place plant_village.py in the project root or pass its path."
        )

    dataset = load_dataset(
        str(script),
        "color",
        trust_remote_code=True,
    )

    if "train" not in dataset or "test" not in dataset:
        raise ValueError(
            f"Expected train and test splits; found {list(dataset.keys())}"
        )

    label_feature = dataset["train"].features.get("label")

    if label_feature is None or not hasattr(label_feature, "names"):
        raise ValueError("The dataset does not expose named class labels.")

    class_names = list(label_feature.names)

    print(f"Dataset loaded: {len(dataset['train'])} training images")
    print(f"Dataset loaded: {len(dataset['test'])} test images")
    print(f"Number of classes: {len(class_names)}")

    return dataset, class_names


def limit_dataset(dataset, limit, seed=42):
    """Return a deterministic, approximately class-balanced subset."""
    import random
    from collections import defaultdict
    from torch.utils.data import Subset

    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset

    # Fast path for Hugging Face datasets.
    try:
        labels = dataset["label"]
    except (TypeError, KeyError, IndexError, AttributeError):
        # Fallback for datasets returning dictionaries or (image, label) tuples.
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            labels.append(
                item["label"] if isinstance(item, dict) else item[1]
            )

    class_indices = defaultdict(list)
    for index, label in enumerate(labels):
        class_indices[int(label)].append(index)

    rng = random.Random(seed)
    for indices in class_indices.values():
        rng.shuffle(indices)

    # Round-robin sampling gives each class a similar number of examples.
    selected = []
    depth = 0
    classes = sorted(class_indices)

    while len(selected) < limit:
        added_this_round = False

        for class_id in classes:
            indices = class_indices[class_id]

            if depth < len(indices):
                selected.append(indices[depth])
                added_this_round = True

                if len(selected) == limit:
                    break

        if not added_this_round:
            break

        depth += 1

    rng.shuffle(selected)
    return Subset(dataset, selected)

def class_counts(hf_split, class_names):
    counts = Counter(int(x) for x in hf_split["label"])
    return {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}

def make_train_validation_split(
    hf_train,
    val_fraction=0.15,
    seed=42,
):
    """Create a stratified training/validation split."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    labels = np.asarray(hf_train["label"], dtype=np.int64)
    indices = np.arange(len(labels))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )

    train_raw = hf_train.select(train_indices.tolist())
    val_raw = hf_train.select(val_indices.tolist())

    return train_raw, val_raw