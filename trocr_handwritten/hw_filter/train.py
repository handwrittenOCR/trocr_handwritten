#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from trocr_handwritten.hw_filter.features import (
    Embedder,
    embedding_table,
    measure_table,
)
from trocr_handwritten.hw_filter.settings import HwFilterSettings
from trocr_handwritten.utils.annotation import register_key
from trocr_handwritten.utils.logging_config import get_logger

logger = get_logger(__name__)

GRID_C = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
GRID_T = np.round(np.arange(0.2, 0.81, 0.05), 2)
CLF = {
    "svm": lambda: make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    ),
    "gboost": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, random_state=42
    ),
    "knn": lambda: make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance"),
    ),
}


def logreg(C):
    """Standardised logistic regression, the retained classifier."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))


def scores(y, p, cls, classes):
    """
    Accuracy and precision, recall, F1 of non manuscrit, overall and per class.

    Args:
        y: True labels (1 = manuscrit).
        p: Predicted labels.
        cls: Region class of each row.
        classes: Classes to report.

    Returns:
        dict: Rounded scores.
    """
    d = {
        "acc": accuracy_score(y, p),
        "p_non_manuscrit": precision_score(y, p, pos_label=0, zero_division=0),
        "r_non_manuscrit": recall_score(y, p, pos_label=0, zero_division=0),
        "f1_non_manuscrit": f1_score(y, p, pos_label=0, zero_division=0),
        "n": len(y),
    }
    d.update(
        {
            f"acc_{c}": accuracy_score(y[cls == c], p[cls == c])
            for c in classes
            if (cls == c).any()
        }
    )
    return {k: v if k == "n" else round(float(v), 4) for k, v in d.items()}


def tune(X, y, groups):
    """
    C and threshold maximising the F1 of non manuscrit on out-of-fold train predictions.

    Args:
        X: Train features.
        y: Train labels.
        groups: Register of each train row (5-fold GroupKFold).

    Returns:
        tuple: (best F1, C, threshold).
    """
    best = (-1, 0.1, 0.5)
    for c in GRID_C:
        p = np.zeros(len(y))
        for a, b in GroupKFold(5).split(X, y, groups):
            p[b] = logreg(c).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
        for t in GRID_T:
            f = f1_score(y, (p >= t).astype(int), pos_label=0)
            if f > best[0]:
                best = (f, c, float(t))
    return best


def main(settings, embeds, clfs, do_tune):
    """
    Train on the train split, score dev and test, save the retained model.

    Baselines: the class rule (Section non manuscrit, the rest manuscrit), and a
    logistic regression and a gradient boosting on image measures. For each
    embedding model, a logistic regression on embedding + class and on embedding +
    measures, and the classifiers of clfs on embedding + class. The model of
    settings.embed_model on embedding + class is saved to out_dir/hw_filter.joblib.

    Args:
        settings: HwFilterSettings.
        embeds: Hugging Face embedding models to compare.
        clfs: Other classifiers to run on embedding + class (keys of CLF).
        do_tune: Choose C and threshold by grouped cross-validation on train.
    """
    out = Path(settings.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = settings.classes
    man = pd.read_csv(settings.manifest)
    man = man[man["class"].isin(classes)].reset_index(drop=True)
    meta = ["region_id", "page", "split", "class", "handwriting"]
    df = man[meta].merge(
        measure_table(man, settings.images_dir, out / "features.parquet"),
        on="region_id",
    )
    onehot = [f"is_{c}" for c in classes]
    for c in classes:
        df[f"is_{c}"] = (df["class"] == c).astype(int)
    cols = [c for c in df.columns if c not in meta]
    df[cols] = df[cols].fillna(0)
    y = (df["handwriting"] == "manuscrit").astype(int).to_numpy()
    tr = (df["split"] == "train").to_numpy()
    groups = (
        df.loc[tr, "page"].map(lambda p: register_key(Path(p).stem + "_0")).to_numpy()
    )

    models = {
        "mesures_logreg": logreg(1.0),
        "mesures_gboost": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=42
        ),
    }
    preds, thr, res = (
        df[["region_id", "split", "class", "handwriting"]].copy(),
        {},
        {"réglages": {}},
    )
    for name, m in models.items():
        m.fit(df.loc[tr, cols], y[tr])
        preds[f"p_{name}"] = m.predict_proba(df[cols])[:, 1]
    for e in embeds:
        embedder = Embedder(e, settings.image_size, settings.batch_size)
        emb = df[["region_id"]].merge(
            embedding_table(
                man,
                settings.images_dir,
                embedder,
                out / f"emb_{embedder.short}.parquet",
            ),
            on="region_id",
        )
        ecols = [c for c in emb.columns if c != "region_id"]
        X = pd.concat([emb[ecols], df[cols].reset_index(drop=True)], axis=1)
        for name, xc in (
            (f"{embedder.short}_classe", ecols + onehot),
            (f"{embedder.short}_mesures", ecols + cols),
        ):
            c = settings.C
            if do_tune:
                f, c, thr[name] = tune(
                    X.loc[tr, xc].reset_index(drop=True), y[tr], groups
                )
                res["réglages"][name] = {
                    "C": c,
                    "seuil": thr[name],
                    "f1_non_manuscrit_cv_train": round(f, 4),
                }
                logger.info(f"{name} {res['réglages'][name]}")
            models[name] = logreg(c).fit(X.loc[tr, xc], y[tr])
            preds[f"p_{name}"] = models[name].predict_proba(X[xc])[:, 1]
        for k in clfs:
            name = f"{embedder.short}_classe_{k}"
            models[name] = CLF[k]().fit(X.loc[tr, ecols + onehot], y[tr])
            preds[f"p_{name}"] = models[name].predict_proba(X[ecols + onehot])[:, 1]
        if e == settings.embed_model:
            name = f"{embedder.short}_classe"
            joblib.dump(
                {
                    "model": models[name],
                    "embed_model": e,
                    "image_size": settings.image_size,
                    "classes": classes,
                    "threshold": thr.get(name, settings.threshold),
                },
                out / "hw_filter.joblib",
            )
            logger.info(f"Modèle retenu ({name}) : {out / 'hw_filter.joblib'}")

    rule = (df["class"] != "Section").astype(int).to_numpy()
    for split in ("dev", "test"):
        s = (df["split"] == split).to_numpy()
        cls = df.loc[s, "class"].to_numpy()
        res[split] = {"regle_classe": scores(y[s], rule[s], cls, classes)}
        res[split].update(
            {
                n: scores(
                    y[s],
                    (preds.loc[s, f"p_{n}"] >= thr.get(n, settings.threshold))
                    .astype(int)
                    .to_numpy(),
                    cls,
                    classes,
                )
                for n in models
            }
        )
    preds.to_csv(out / "predictions.csv", index=False)
    (out / "scores.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for split in ("dev", "test"):
        logger.info(f"{split}\n{pd.DataFrame(res[split]).T.to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the handwriting filter on the reviewed regions"
    )
    defaults = HwFilterSettings()
    parser.add_argument("--manifest", default=defaults.manifest)
    parser.add_argument("--images-dir", default=defaults.images_dir)
    parser.add_argument("--out-dir", default=defaults.out_dir)
    parser.add_argument(
        "--embed-model",
        default=defaults.embed_model,
        help="Embedding model of the saved filter.",
    )
    parser.add_argument(
        "--embed",
        nargs="*",
        default=None,
        help="Embedding models to compare (default: --embed-model only).",
    )
    parser.add_argument("--clf", nargs="*", default=[], choices=list(CLF))
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    settings = HwFilterSettings(
        manifest=args.manifest,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        embed_model=args.embed_model,
    )
    main(
        settings,
        args.embed if args.embed is not None else [settings.embed_model],
        args.clf,
        args.tune,
    )
