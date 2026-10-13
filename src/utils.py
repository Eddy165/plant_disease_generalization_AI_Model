from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(data, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pathlib import Path
import torch


def save_checkpoint(model, optimizer, epoch, best_val_accuracy, path, **extra):
    """Save model and training state to a checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "best_val_accuracy": best_val_accuracy,
            **extra,
        },
        path,
    )



def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    """Load a checkpoint, optionally restoring model and optimizer states."""
    from pathlib import Path

    # Support load_checkpoint(path, device) as well as
    # load_checkpoint(path, model, optimizer=...).
    if isinstance(model, (str, torch.device)):
        map_location = model
        model = None

    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )

    if model is not None:
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint),
        )
        model.load_state_dict(state_dict)

    if (
        optimizer is not None
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint