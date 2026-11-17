# Handwriting filter

Classifies a region crop of the transcribed classes (Plein Texte, Marge, Nom, Section) as handwritten or not, to decide which regions to transcribe. A region is handwritten if it has no printed text and at least one handwritten word, or printed text and at least two handwritten words.

Model: frozen DINOv2-small embedding of the crop (padded to a square, 448 px) plus the region class (one-hot), standardised logistic regression (C = 0.1, threshold 0.5). Trained on the reviewed statuses of `pages_manifest.csv`, splits disjoint by register.

On the test split (1 049 regions, statuses reviewed): accuracy 0.990, non-handwritten precision 0.985, recall 0.987, F1 0.986. The class rule (Section non-handwritten, the rest handwritten) gives 0.874.

## 1. Train and evaluate

```bash
python -m trocr_handwritten.hw_filter.train --out-dir data/processed/hw_filter
```

Reads `data/processed/pages_manifest.csv` and the pages of `data/raw/initial/`, caches measures and embeddings as parquet in `--out-dir`, trains on train, scores dev and test against the class rule and image-measure baselines, and writes `scores.json`, `predictions.csv` and the retained model `hw_filter.joblib`.

Comparisons: `--embed facebook/dinov2-small microsoft/resnet-18 microsoft/trocr-small-handwritten` (other frozen encoders), `--clf svm gboost knn` (other classifiers on the same embedding), `--tune` (C and threshold chosen by 5-fold cross-validation on train, grouped by register). None of them beat the default.

## 2. Apply to new pages

Cut the regions with the layout parser, then classify its crops:

```bash
python -m trocr_handwritten.parse.layout_parser <pages> --output <crops> --device cpu
python -m trocr_handwritten.hw_filter.predict <crops> --model data/processed/hw_filter/hw_filter.joblib --output <crops>/handwriting.csv
```

`handwriting.csv` has one row per crop of the four classes: page, class, crop, `p_manuscrit`, `handwriting` (`manuscrit` / `non_manuscrit`). Crops classified `manuscrit` go to transcription; uncertain ones (probability near the threshold) are worth a manual check.
