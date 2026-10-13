# PlantDoc external dataset setup

Use the dataset's official repository or project page and review its license and label definitions before downloading. Keep raw data outside version control.

1. Download/extract PlantDoc into `data/plantdoc_raw/`.
2. Inspect its directory structure and labels.
3. Create a documented mapping table from source labels to PlantVillage class names.
4. Exclude ambiguous or unmatched classes. Do not relabel by guesswork.
5. Build a mapped external evaluation set. The baseline `src/cross_dataset.py` currently expects every checkpoint class directory, so for a practical subset evaluation, adapt the evaluator to an explicit label subset and report that subset; do not fill missing classes with fabricated data.
6. Keep all external images separate from training and validation data.
