
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from .data import build_transforms
from .models import build_model
from .utils import ensure_dir, get_device, save_json


# Only use categories with a defensible label correspondence.
# Ambiguous categories are deliberately excluded.
PLANTDOC_TO_MODEL = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "grape leaf black rot": "Grape___Black_rot",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a PlantVillage-trained model "
            "on mapped PlantDoc classes."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data_dir)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not data_path.is_dir():
        raise NotADirectoryError(
            f"Dataset directory not found: {data_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    model_class_to_id = {
        name: index for index, name in enumerate(class_names)
    }

    unknown_targets = sorted(
        set(PLANTDOC_TO_MODEL.values()) - set(class_names)
    )
    if unknown_targets:
        raise ValueError(
            "Mapped model classes are missing from checkpoint: "
            f"{unknown_targets}"
        )

    dataset = datasets.ImageFolder(
        str(data_path),
        transform=build_transforms(
            checkpoint.get("image_size", 224),
            False,
        ),
    )

    available_folders = set(dataset.classes)

    active_mapping = {
        folder: target
        for folder, target in PLANTDOC_TO_MODEL.items()
        if folder in available_folders
    }

    if not active_mapping:
        raise ValueError(
            "No mapped PlantDoc folders were found. Check the data "
            "directory and exact folder names."
        )

    # Preserve the original image count before filtering.
    original_image_count = len(dataset.samples)

    # Map selected PlantDoc labels to the checkpoint's class indices.
    selected_samples = []

    for image_path, original_target in dataset.samples:
        folder = dataset.classes[original_target]

        if folder in active_mapping:
            model_target = model_class_to_id[
                active_mapping[folder]
            ]
            selected_samples.append(
                (image_path, model_target)
            )

    if not selected_samples:
        raise ValueError(
            "No images were found in the mapped categories."
        )

    excluded_folders = sorted(
        available_folders - set(active_mapping)
    )

    dataset.samples = selected_samples
    dataset.imgs = selected_samples
    dataset.targets = [
        target for _, target in selected_samples
    ]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = get_device()

    model = build_model(
        checkpoint["model_name"],
        checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions)

    # Classes present in the ground-truth evaluation subset.
    evaluated_ids = sorted(set(y_true))
    evaluated_names = [
        class_names[class_id] for class_id in evaluated_ids
    ]
    evaluated_id_set = set(evaluated_ids)

    # Generate a report for the same classes used by the macro metrics.
    report = classification_report(
        y_true,
        y_pred,
        labels=evaluated_ids,
        target_names=evaluated_names,
        output_dict=True,
        zero_division=0,
    )

    # A prediction outside evaluated_ids is incorrect for this subset.
    predictions_outside_evaluated_classes = sum(
        prediction not in evaluated_id_set
        for prediction in y_pred
    )

    # All three macro metrics use exactly the evaluated class IDs.
    metrics = {
        "dataset": str(data_path),
        "checkpoint": str(checkpoint_path),
        "num_images_available": original_image_count,
        "num_images_evaluated": len(y_true),
        "num_images_excluded": (
            original_image_count - len(y_true)
        ),
        "num_plantdoc_folders": len(available_folders),
        "num_mapped_folders": len(active_mapping),
        "num_model_classes_evaluated": len(evaluated_ids),
        "mapped_class_coverage_of_model": (
            len(evaluated_ids) / len(class_names)
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=evaluated_ids,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=evaluated_ids,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=evaluated_ids,
            average="macro",
            zero_division=0,
        ),
        "predictions_outside_evaluated_classes": (
            predictions_outside_evaluated_classes
        ),
        "class_mapping": active_mapping,
        "excluded_plantdoc_folders": excluded_folders,
        "evaluation_note": (
            "Results apply only to mapped PlantDoc categories. "
            "This is external evaluation, not training or fine-tuning. "
            "Predictions outside the evaluated class subset count "
            "as incorrect."
        ),
    }

    output_dir = ensure_dir("artifacts/cross_dataset")

    save_json(
        metrics,
        output_dir / "metrics.json",
    )

    pd.DataFrame(report).T.to_csv(
        output_dir / "classification_report.csv",
        index=True,
        index_label="class",
    )

    print(json.dumps(metrics, indent=2))
    print(
        "\nClassification report: "
        f"{output_dir / 'classification_report.csv'}"
    )
    print(f"Metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
