#!/usr/bin/env python3
import sys
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}


def count_images_in_each_class(path: Path) -> OrderedDict:
    counts = OrderedDict()
    subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(f"No subdirectories found in {path}")
    for subdir in subdirs:
        counts[subdir.name] = sum(
            1 for f in subdir.iterdir() if f.is_file() and f.suffix in IMG_EXTS
        )
    return counts


def distribute(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")
    plant_type = path.name
    counts = count_images_in_each_class(path)
    if not any(counts.values()):
        raise ValueError(f"No images found in class subdirectories of {path}")

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{plant_type} class distribution")
    ax_pie.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
    ax_bar.bar(counts.keys(), counts.values())
    plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")
    ax_bar.set_ylabel("Number of images")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./Distribution.py <path_to_dataset>")
        sys.exit(1)
    try:
        distribute(sys.argv[1])
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
