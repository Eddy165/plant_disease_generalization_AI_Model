from __future__ import annotations
import argparse, json, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from .data import load_plantvillage, HFDatasetAdapter, build_transforms
from .models import build_model
from .utils import get_device, ensure_dir, save_json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw, class_names = load_plantvillage()
    class_names = ckpt.get("class_names", class_names)
    ds = HFDatasetAdapter(raw["test"], build_transforms(ckpt.get("image_size", 224), False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    device = get_device()
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    ys, preds = [], []
    latencies = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) / len(x))
            ys.extend(y.tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
    report = classification_report(ys, preds, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(ys, preds),
        "macro_precision": precision_score(ys, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(ys, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "mean_batch_per_image_latency_ms": float(np.mean(latencies) * 1000),
        "num_test_images": len(ys),
        "model_size_mb": Path(args.checkpoint).stat().st_size / (1024 * 1024),
    }
    out = ensure_dir("artifacts/evaluation")
    save_json(metrics, out / "metrics.json")
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    cm = confusion_matrix(ys, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160); plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"Saved detailed report to {out}")

if __name__ == "__main__":
    main()
