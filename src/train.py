from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    HFDatasetAdapter,
    build_transforms,
    limit_dataset,
    load_plantvillage,
    make_train_validation_split,
)
from .models import MODEL_NAMES, build_model
from .utils import ensure_dir, get_device, load_checkpoint, save_json, seed_everything


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or evaluation epoch and return sample-weighted metrics."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    y_true, y_pred = [], []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in tqdm(
            loader,
            leave=False,
            desc="Training" if training else "Evaluating",
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("The data loader returned no samples.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a plant-disease classifier on PlantVillage."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="mobilenet_v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the classifier head.")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Lower learning rate for pretrained backbone.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="0 means use the complete training split.")
    parser.add_argument("--max-val-samples", type=int, default=0,
                        help="0 means use the complete validation split.")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="0 means use the complete official test split.")
    parser.add_argument("--pretrained", choices=["true", "false"], default="true")
    parser.add_argument("--dataset-script", default="./plant_village.py")
    parser.add_argument("--patience", type=int, default=3,
                        help="Reduce LR after this many epochs without validation F1 improvement.")

    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    if args.lr <= 0 or args.backbone_lr <= 0:
        parser.error("--lr and --backbone-lr must be greater than 0")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    return args


def make_optimizer(model, model_name, head_lr, backbone_lr, pretrained):
    """Use a smaller LR for pretrained feature extractors than for the new head."""
    if model_name == "mobilenet_v2":
        head_params = list(model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    elif model_name == "resnet50":
        head_params = list(model.fc.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    else:
        # Custom CNN has no pretrained ImageNet backbone/head split.
        return torch.optim.AdamW(
            model.parameters(), lr=head_lr, weight_decay=1e-4
        )

    if pretrained:
        groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    else:
        groups = [{"params": model.parameters(), "lr": head_lr}]

    return torch.optim.AdamW(groups, weight_decay=1e-4)


def main():
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    use_pretrained = args.pretrained == "true"

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Pretrained ImageNet weights: {use_pretrained}")
    print(f"Seed: {args.seed}")

    # 1. Load the official dataset and its canonical label names.
    raw, class_names = load_plantvillage(args.dataset_script)
    if len(class_names) < 2:
        raise ValueError(f"Expected multiple classes, found {len(class_names)}.")

    # 2. Split only the official training split. Keep official test data separate.
    train_raw, val_raw = make_train_validation_split(
        raw["train"], val_fraction=args.val_fraction, seed=args.seed
    )

    # 3. Apply sample limits to raw Hugging Face splits BEFORE image adapters.
    # This avoids decoding/transformation of images just to select sample indices.
    train_raw = limit_dataset(
        train_raw, args.max_train_samples, seed=args.seed
    )
    val_raw = limit_dataset(
        val_raw, args.max_val_samples, seed=args.seed + 1
    )
    test_raw = limit_dataset(
        raw["test"], args.max_test_samples, seed=args.seed + 2
    )

    # 4. Use augmentation only for training; validation/test remain deterministic.
    train_ds = HFDatasetAdapter(
        train_raw, build_transforms(args.image_size, train=True)
    )
    val_ds = HFDatasetAdapter(
        val_raw, build_transforms(args.image_size, train=False)
    )
    test_ds = HFDatasetAdapter(
        test_raw, build_transforms(args.image_size, train=False)
    )

    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise ValueError("Train, validation, and test datasets must be non-empty.")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,  # Safe default for Windows.
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_options)

    print(
        f"Samples — train: {len(train_ds)}, validation: {len(val_ds)}, "
        f"test: {len(test_ds)}; classes: {len(class_names)}"
    )
    if args.max_test_samples > 0 and args.max_test_samples < len(raw["test"]):
        print("WARNING: This run uses a test subset; final metrics require the full test set.")

    # 5. Build model. For pretrained models, use a small backbone LR and larger head LR.
    model = build_model(
        args.model, num_classes=len(class_names), pretrained=use_pretrained
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(
        model, args.model, args.lr, args.backbone_lr, use_pretrained
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.patience
    )

    output_dir = ensure_dir("artifacts")
    checkpoint_path = output_dir / f"{args.model}_best.pt"
    history_path = output_dir / f"{args.model}_training_history.json"

    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    start_time = time.time()

    # 6. Select checkpoints by validation macro-F1, never by test performance.
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["macro_f1"])

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}, "
            f"macro-F1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            checkpoint = {
                "model_name": args.model,
                "num_classes": len(class_names),
                "class_names": class_names,
                "image_size": args.image_size,
                "state_dict": model.state_dict(),
                "seed": args.seed,
                "pretrained": use_pretrained,
                "best_validation_macro_f1": best_val_f1,
                "epoch": epoch,
            }
            temp_path = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, temp_path)
            temp_path.replace(checkpoint_path)
            print(f"Saved best checkpoint (epoch {epoch}).")

        save_json(
            {
                "model": args.model,
                "device": str(device),
                "pretrained": use_pretrained,
                "epochs_requested": args.epochs,
                "seed": args.seed,
                "train_samples": len(train_ds),
                "validation_samples": len(val_ds),
                "test_samples": len(test_ds),
                "elapsed_seconds": time.time() - start_time,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_val_f1,
                "history": history,
            },
            history_path,
        )

    # 7. Reload the best checkpoint before the held-out test evaluation.
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    test_metrics = run_epoch(model, test_loader, criterion, device)

    summary = {
        "model": args.model,
        "device": str(device),
        "pretrained": use_pretrained,
        "epochs_requested": args.epochs,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "test_samples": len(test_ds),
        "test_is_limited": (
            args.max_test_samples > 0
            and args.max_test_samples < len(raw["test"])
        ),
        "elapsed_seconds": time.time() - start_time,
        "best_epoch": checkpoint.get("epoch"),
        "best_validation_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
    }
    save_json(summary, history_path)

    print("\nTraining complete.")
    print(f"Best epoch: {checkpoint.get('epoch')}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Test metrics: {test_metrics}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
