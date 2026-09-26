#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Transformation import build_mask, apply_mask
from train import extract_features


def load_model(learnings_dir):
    with open(os.path.join(learnings_dir, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    with open(os.path.join(learnings_dir, "class_names.json")) as f:
        class_names = json.load(f)
    return bundle["model"], bundle["scaler"], class_names


def main():
    parser = argparse.ArgumentParser(
        description="Leaffliction — predict a leaf's disease class (Part 4, sklearn)."
    )
    parser.add_argument("image", help="Path to a leaf image.")
    parser.add_argument("-learnings", dest="learnings_dir", default="learnings",
                         help="Directory produced by train.py (default: learnings/)")
    args = parser.parse_args()

    clf, scaler, class_names = load_model(args.learnings_dir)

    original = cv2.imread(args.image)
    if original is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    mask = build_mask(original)
    masked = apply_mask(original, mask)

    feats = extract_features(original).reshape(1, -1)
    feats_s = scaler.transform(feats)

    probs = clf.predict_proba(feats_s)[0]
    idx = int(np.argmax(probs))
    predicted_class = clf.classes_[idx]
    confidence = float(probs[idx])

    print(f"Class predicted : {predicted_class}  ({confidence * 100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Masked")
    axes[1].axis("off")
    fig.suptitle(f"Class predicted : {predicted_class}")
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.image)) or "."
    out_path = os.path.join(out_dir, "prediction.png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved prediction figure to {out_path}")


if __name__ == "__main__":
    main()