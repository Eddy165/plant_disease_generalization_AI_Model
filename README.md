# Plant Disease Classification with Cross-Dataset Generalization Analysis

A research-oriented deep-learning project comparing a custom CNN, ResNet50, and MobileNetV2 for plant-leaf disease classification. It includes dataset inspection, reproducible training, metrics, cross-dataset evaluation hooks, Grad-CAM explanations, uncertainty warnings, and a Streamlit demo.

> **Research integrity:** this repository is a runnable project scaffold, not a claim that models have already been trained or that field performance is established. Results must be generated on your machine/runtime and reported honestly. PlantVillage-to-PlantDoc label mapping is not automatically assumed to be equivalent.

## Features

- PyTorch models: `custom_cnn`, `resnet50`, `mobilenet_v2`
- PlantVillage loading via Hugging Face `datasets`
- Leakage-aware split support using leaf identifiers when available
- Accuracy, macro precision/recall/F1, per-class report, confusion matrix
- Cross-dataset evaluation command for a directory arranged by class
- Grad-CAM heatmaps for supported convolutional models
- Confidence threshold warning in the Streamlit application
- Model size and approximate single-image inference latency reporting
- Configurable training, checkpoints, and CSV/JSON outputs

## Project structure

```text
plant_disease_generalization/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── configs/default.yaml
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── utils.py
│   └── cross_dataset.py
├── scripts/
│   ├── inspect_dataset.py
│   └── download_plantdoc_instructions.md
├── tests/test_models.py
├── artifacts/                 # generated checkpoints/results
└── data/                      # local datasets; do not commit large data
```

## 1. Setup

Python 3.10 or 3.11 is recommended. GPU training is optional but strongly recommended.

### Local Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Google Colab

```python
%pip install -r requirements.txt
```

Upload or clone this project into the Colab runtime, then check:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## 2. Inspect PlantVillage

The default loader uses the Hugging Face dataset `mohanty/PlantVillage`, configuration `color`.

```bash
python scripts/inspect_dataset.py
```

This downloads the dataset and writes a class-count CSV plus sample grid under `artifacts/inspection/`.

For reproducibility, save the dataset revision / retrieval date and exact split details in your report. Dataset versions and available columns may change.

## 3. Train models

Example commands:

```bash
python -m src.train --model custom_cnn --epochs 10 --batch-size 32
python -m src.train --model resnet50 --epochs 8 --batch-size 32
python -m src.train --model mobilenet_v2 --epochs 8 --batch-size 32
```

For a quick smoke test:

```bash
python -m src.train --model custom_cnn --epochs 1 --max-train-samples 800 --max-test-samples 200 --batch-size 16
```

Training downloads ImageNet weights for transfer-learning models if they are not cached. If you need fully offline training, use `--pretrained false`.

Checkpoints and reports are saved in `artifacts/`. The training script uses the dataset's existing train/test split where available. Do not tune model choices or thresholds on the test set. If the source dataset has leaf identifiers, the inspection script reports them; verify grouping and split integrity before making research claims.

## 4. Evaluate a checkpoint

```bash
python -m src.evaluate --checkpoint artifacts/mobilenet_v2_best.pt
```

Outputs include JSON metrics, a per-class classification report CSV, and a confusion-matrix image.

## 5. Cross-dataset generalization

Prepare an independently sourced field dataset as:

```text
data/plantdoc_mapped/
├── Tomato___Early_blight/
│   ├── image1.jpg
│   └── image2.jpg
└── Tomato___healthy/
    └── image3.jpg
```

The folder names must exactly match the training class names for this baseline. Only include classes with defensible label equivalence. PlantDoc labels may differ in crop/disease wording and taxonomy; document every mapping and exclude ambiguous classes rather than silently relabeling them.

```bash
python -m src.cross_dataset --checkpoint artifacts/mobilenet_v2_best.pt --data-dir data/plantdoc_mapped
```

This evaluation is only meaningful if the external images are genuinely independent, labels are compatible, and preprocessing is identical. Report the evaluated class subset and number of images per class.

## 6. Run the app

```bash
streamlit run app.py
```

Choose a checkpoint, upload a leaf image, and review the predicted class, confidence, Grad-CAM overlay, and threshold warning. A high softmax score is not proof that a prediction is correct. The warning threshold is a simple heuristic, not a calibrated probability guarantee.

## 7. Research protocol

1. Record package versions, dataset revision, random seed, hardware, and preprocessing.
2. Use identical data splits and training/evaluation transforms for all three models.
3. Train the custom CNN from scratch; compare transfer learning with pretrained ResNet50 and MobileNetV2.
4. Select checkpoints and tune hyperparameters using validation data, not test data.
5. Report accuracy, macro F1, per-class precision/recall, confusion matrices, model size, and latency.
6. For cross-dataset evaluation, freeze the model and preprocessing before evaluating the external set.
7. Consider Expected Calibration Error (ECE) and risk-coverage curves if implementing calibration/rejection.
8. Discuss limitations: class mismatch, dataset shift, class imbalance, duplicate/near-duplicate images, and the difference between image classification and field diagnosis.

## Important limitations

- The model is an educational/research aid, not a substitute for an agronomist.
- Predictions are only as reliable as training data and label quality.
- Grad-CAM is a qualitative visualization, not proof of causal reasoning.
- Cross-dataset results are not available until you run the external evaluation.
- Large datasets and trained weights are intentionally excluded from this repository.
