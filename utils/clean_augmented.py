import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Augmentation import transform_registry

# matches "_Flip", "_Flip2", "_Rotate", "_Rotate3", etc. at the end of the stem
AUGMENT_STEM_RE = re.compile(rf"_({'|'.join(transform_registry)})\d*$")


def is_augmented(path: Path) -> bool:
    return bool(AUGMENT_STEM_RE.search(path.stem))


def clean(path_str: str) -> int:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")
    removed = 0
    for file in path.rglob("*"):
        if file.is_file() and is_augmented(file):
            print(f"Removing {file}")
            file.unlink()
            removed += 1
    return removed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 utils/clean_augmented.py <directory>")
        sys.exit(1)
    try:
        removed = clean(sys.argv[1])
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Removed {removed} augmented image(s).")
