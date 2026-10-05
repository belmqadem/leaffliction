#!/usr/bin/env python3
import random
import itertools
import re
import shutil
import sys
from pathlib import Path
from Augmentation import transform_registry
from Distribution import IMG_EXTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# matches "_Flip", "_Flip2", "_Rotate", "_Rotate3", etc. at the end of the stem
AUGMENT_STEM_RE = re.compile(rf"_({'|'.join(transform_registry)})\d*$")


def is_original_image(f: Path) -> bool:
    return f.is_file() and f.suffix in IMG_EXTS \
        and not AUGMENT_STEM_RE.search(f.stem)


def balance_dataset(root_str: str, seed: int = 42) -> None:
    root = Path(root_str)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    random.seed(seed)

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {root}")

    augmented_root = root.parent / "augmented_directory" / root.name
    augmented_root.mkdir(parents=True, exist_ok=True)
    print(f"Building balanced dataset in {augmented_root}")

    copy_dirs = []
    for class_dir in class_dirs:
        dst_dir = augmented_root / class_dir.name
        dst_dir.mkdir(exist_ok=True)
        for f in class_dir.iterdir():
            if is_original_image(f):
                dst = dst_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
        copy_dirs.append(dst_dir)

    class_originals = {
        d: sorted([f for f in d.iterdir() if is_original_image(f)])
        for d in copy_dirs
    }
    class_totals = {
        d: sum(1 for f in d.iterdir() if f.is_file() and f.suffix in IMG_EXTS)
        for d in copy_dirs
    }

    counts = {d: len(files) for d, files in class_originals.items()}
    target = max(counts.values())

    for class_dir, originals in class_originals.items():
        total_existing = class_totals[class_dir]
        deficit = target - total_existing
        if deficit <= 0:
            print(f"{class_dir.name}: already balanced ({total_existing})")
            continue
        if not originals:
            raise ValueError(
                f"{class_dir.name}: has 0 original images, cannot generate "
                f"augmentations to reach the target of {target}"
            )

        print(
            f"{class_dir.name}: generating {deficit} images "
            f"({total_existing} -> {target})"
        )

        # every (image, transform_name) pair, shuffled, so we exhaust
        # combos before ever reusing one, and spread transforms evenly
        # across source images instead of hammering the first few
        combos = list(itertools.product(originals, transform_registry.keys()))
        random.shuffle(combos)

        generated = 0
        combo_iter = itertools.cycle(combos)
        uses = {}

        while generated < deficit:
            img_path, transform_name = next(combo_iter)
            key = (img_path, transform_name)
            uses[key] = uses.get(key, 0) + 1
            n = uses[key]
            tag = transform_name if n == 1 else f"{transform_name}{n}"

            out_name = f"{img_path.stem}_{tag}{img_path.suffix}"
            out_path = class_dir / out_name
            if out_path.exists():
                continue  # don't overwrite/recount an existing augmentation

            transform_fn = transform_registry[transform_name]
            transform_fn(img_path, tag=tag)  # writes to disk under `tag`
            generated += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./utils/balance_dataset.py <path_to_dataset_root>")
        sys.exit(1)
    try:
        balance_dataset(sys.argv[1])
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
