from datasets import load_dataset
from collections import Counter

def main():
    print("Loading PlantVillage (color) using local loader...")

    ds = load_dataset(
        "./plant_village.py",
        "color",
        trust_remote_code=True,
    )

    print("\nDataset splits:")
    for split_name, split in ds.items():
        print(f"{split_name}: {len(split):,} images")
        print("Columns:", split.column_names)

    train = ds["train"]
    label_feature = train.features.get("label")

    if label_feature is None or not hasattr(label_feature, "names"):
        raise RuntimeError(
            "Expected a ClassLabel feature named 'label'. "
            f"Actual features: {train.features}"
        )

    class_names = label_feature.names
    counts = Counter(train["label"])

    print(f"\nNumber of classes: {len(class_names)}")
    print("\nTraining images per class:")
    for label_id, class_name in enumerate(class_names):
        print(f"{class_name}: {counts.get(label_id, 0):,}")

    print("\nDataset inspection completed successfully.")

if __name__ == "__main__":
    main()
