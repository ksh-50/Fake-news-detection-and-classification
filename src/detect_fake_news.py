from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_clean import clean_text


def load_pipeline_or_parts(pipeline_path, model_path, vectorizer_path):
    if pipeline_path:
        return joblib.load(pipeline_path), None, None
    if not (model_path and vectorizer_path):
        raise ValueError("Provide --pipeline OR both --model and --vectorizer.")
    return None, joblib.load(model_path), joblib.load(vectorizer_path)


def main():
    ap = argparse.ArgumentParser(description="Detect fake news for a single text.")
    ap.add_argument("--pipeline", help="Path to pipeline.joblib")
    ap.add_argument("--model", help="Path to model.joblib")
    ap.add_argument("--vectorizer", help="Path to vectorizer.joblib")
    ap.add_argument("--text", required=True)
    ap.add_argument("--threshold", type=float, default=0.40)
    args = ap.parse_args()

    pipe, clf, vec = load_pipeline_or_parts(args.pipeline, args.model, args.vectorizer)
    s = clean_text(args.text)

    if pipe is not None:
        prob = float(pipe.predict_proba([s])[0, 1])
    else:
        X = vec.transform([s])
        prob = float(clf.predict_proba(X)[0, 1])

    label = "FAKE" if prob >= args.threshold else "REAL"
    print(f"Label: {label} | Fake probability: {prob:.3f} | Threshold: {args.threshold:.2f}")


if __name__ == "__main__":
    main()
