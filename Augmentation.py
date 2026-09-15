#!/usr/bin/env python3
import random
import sys
from pathlib import Path
import cv2
import numpy as np


def flip_img(path: Path, tag: str = "Flip"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    fliped_img = cv2.flip(img, 1)
    fliped_img_name = f"{path.stem}_{tag}{path.suffix}"
    cv2.imwrite(path.with_name(fliped_img_name), fliped_img)
    return fliped_img


def rotate_img(path: Path, tag: str = "Rotate"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    rotated_img_name = f"{path.stem}_{tag}{path.suffix}"
    cv2.imwrite(path.with_name(rotated_img_name), rotated_img)
    return rotated_img


def skew_img(path: Path, tag: str = "Skew"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    # 4 coordinates in original img
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # 4 coordinates in skewed (shift top-left and top-right in)
    pts2 = np.float32([[50, 0], [w - 50, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    skewed_img = cv2.warpPerspective(img, M, (w, h))
    skewed_img_name = f"{path.stem}_{tag}{path.suffix}"
    cv2.imwrite(path.with_name(skewed_img_name), skewed_img)
    return skewed_img


def shear_img(path: Path, tag: str = "Shear"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    # 0.2 slants the X axis based on Y height
    shear_factor = 0.2
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared_img = cv2.warpAffine(img, M, (w + int(h * shear_factor), h))
    sheared_img_name = f"{path.stem}_{tag}{path.suffix}"
    cv2.imwrite(path.with_name(sheared_img_name), sheared_img)
    return sheared_img


def crop_img(path: Path, tag: str = "Crop"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    margin_h, margin_w = int(h * 0.1), int(w * 0.1)
    cropped_img = img[margin_h : h - margin_h, margin_w : w - margin_w]
    cropped_img_name = f"{path.stem}_{tag}{path.suffix}"
    cv2.imwrite(path.with_name(cropped_img_name), cropped_img)
    return cropped_img


def distort_img(path: Path, tag: str = "Distortion"):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    # mapping matrices
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    # distort x coordinates using a sine wave
    for i in range(h):
        map_x[i, :] = map_x[i, :] + 10.0 * np.sin(map_y[i, :] / 20.0)
    distorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    distorted_img_name = f"{path.stem}_{tag}{path.suffix}"
    cv2.imwrite(path.with_name(distorted_img_name), distorted_img)
    return distorted_img


transform_registry = {
    "Flip": flip_img,
    "Rotate": rotate_img,
    "Skew": skew_img,
    "Shear": shear_img,
    "Crop": crop_img,
    "Distortion": distort_img,
}


def augment(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    for transform in transform_registry.values():
        transform(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py <path_to_image>")
        sys.exit(1)
    try:
        augment(sys.argv[1])
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
