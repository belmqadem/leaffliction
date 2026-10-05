#!/usr/bin/env python3


import argparse
import json
import os
import pickle
import random
import shutil
import sys
import zipfile
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Transformation import build_mask, apply_mask

SEED = 42
VAL_SPLIT = 0.2
MIN_VAL_IMAGES = 100


def list_class_dirs(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def collect_images(root, class_dirs):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    samples = []  # (path, label)
    for label in class_dirs:
        class_path = os.path.join(root, label)
        for fname in os.listdir(class_path):
            if fname.endswith(exts):
                samples.append((os.path.join(class_path, fname), label))
    return samples


def extract_features(img):
    """Color + shape feature vector from the masked leaf. ~20 dims, fast."""
    mask = build_mask(img)
    masked = apply_mask(img, mask)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)

    feats = []
    fg = mask > 0
    for space in (masked, hsv, lab):
        for ch in range(3):
            values = space[:, :, ch][fg]
            if values.size == 0:
                values = np.array([0.0])
            feats.append(float(values.mean()))
            feats.append(float(values.std()))

    area = float(np.sum(fg))
    total = float(mask.size)
    feats.append(area)
    feats.append(area / total if total else 0.0)
    return np.array(feats, dtype=np.float32)


def _extract_one(args):
    path, label = args
    img = cv2.imread(path)
    if img is None:
        return None
    try:
        return extract_features(img), label, path
    except Exception as exc:
        print(f"  (skipped {path}: {exc})")
        return None


def build_dataset(samples, workers):
    with Pool(processes=workers) as pool:
        results = pool.map(_extract_one, samples)
    results = [r for r in results if r is not None]
    X = np.array([r[0] for r in results], dtype=np.float32)
    labels = [r[1] for r in results]
    return X, labels


def make_zip(out_dir, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arc = os.path.relpath(full, os.path.dirname(out_dir))
                zf.write(full, arc)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Leaffliction — train a leaf-disease classifier (Part 4, sklearn)."
        )
    )
    parser.add_argument(
        "directory", help="Root directory containing one subdir per class."
    )
    parser.add_argument(
        "-out",
        dest="out_dir",
        default="learnings",
        help="Where to save the model + dataset copy (default: learnings/)",
    )
    parser.add_argument(
        "-workers",
        type=int,
        default=max(cpu_count() - 1, 1),
        help="Parallel workers for feature extraction.",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    class_dirs = list_class_dirs(args.directory)
    if not class_dirs:
        print(
            f"No class subdirectories found in {args.directory}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    samples = collect_images(args.directory, class_dirs)
    print(f"Found {len(samples)} images total. Extracting features")

    X, labels = build_dataset(samples, args.workers)
    print(f"Extracted features for {len(X)} images.")

    n_val = max(int(len(X) * VAL_SPLIT), MIN_VAL_IMAGES)
    if n_val >= len(X):
        print(
            "Not enough images to reach the minimum validation set size.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=n_val, random_state=SEED, stratify=labels
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1
    )
    clf.fit(X_train_s, y_train)

    val_pred = clf.predict(X_val_s)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_acc * 100:.2f}%  (n={len(X_val)})")
    if val_acc < 0.90:
        print("WARNING: validation accuracy is below the required 90%.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "model.pkl"), "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler}, f)
    with open(os.path.join(args.out_dir, "class_names.json"), "w") as f:
        json.dump(sorted(set(labels)), f)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"val_accuracy": val_acc, "val_size": len(X_val)}, f)

    dataset_copy = os.path.join(args.out_dir, "dataset")
    if os.path.exists(dataset_copy):
        shutil.rmtree(dataset_copy)
    shutil.copytree(args.directory, dataset_copy)

    zip_path = f"{args.out_dir}.zip"
    make_zip(args.out_dir, zip_path)
    print(f"Saved model + dataset to {args.out_dir}/, zipped as {zip_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./train.py <path_to_dataset_root> \
              -out <output_directory> -workers <workers_number> ")
        sys.exit(1)
    try:
        main()
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
