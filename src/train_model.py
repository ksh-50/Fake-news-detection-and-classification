from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

LABELS: Final[Tuple[str, str]] = ("REAL", "FAKE")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_any(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin-1")


def pick_text_column(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    for alt in ("combined_text", "text", "content", "article", "body"):
        if alt in df.columns:
            return alt
    if "title" in df.columns:
        return "title"
    raise ValueError(f"No suitable text column. Available: {list(df.columns)[:20]}")


def plot_confusion_matrix(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(LABELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curve(x, y, out, title, xlabel, ylabel) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Fake News Detector — TF-IDF + Logistic Regression"
    )
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--outdir", default="outputs")
    a = ap.parse_args()

    outdir = ensure_dir(Path(a.outdir))
    charts = ensure_dir(outdir / "charts")

    df_real = read_csv_any(Path(a.real))
    df_fake = read_csv_any(Path(a.fake))

    for df in (df_real, df_fake):
        title = df["title"].fillna("") if "title" in df.columns else ""
        txt = df.get(a.text_col, df.get("text", "")).fillna("")
        df["combined_text"] = (title + " " + txt).str.strip()

    X = pd.concat([df_real["combined_text"], df_fake["combined_text"]], ignore_index=True)
    y = np.array([0] * len(df_real) + [1] * len(df_fake))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=3,
            max_features=20_000,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"CV F1 (5-fold): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "cv_f1_macro_mean": float(cv_f1.mean()),
        "cv_f1_macro_std": float(cv_f1.std()),
        "report": classification_report(y_test, y_pred, target_names=LABELS, output_dict=True),
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, charts / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_curve(fpr, tpr, charts / "roc_curve.png", "ROC Curve", "FPR", "TPR")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plot_curve(rec, prec, charts / "pr_curve.png", "Precision-Recall", "Recall", "Precision")

    joblib.dump(pipe, outdir / "pipeline.joblib")
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    joblib.dump(vec, outdir / "vectorizer.joblib")
    joblib.dump(clf, outdir / "model.joblib")

    print("Done.", str(outdir.resolve()))
    print(json.dumps({k: metrics[k] for k in ["accuracy", "roc_auc", "avg_precision"]}, indent=2))


if __name__ == "__main__":
    main()
