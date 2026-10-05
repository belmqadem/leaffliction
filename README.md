# Leaffliction

## 1. Project overview

Leaffliction is a 42 school computer vision project that classifies leaf
diseases from images. The pipeline has five steps: analyse the class
distribution of a dataset, augment images, extract plant features with
PlantCV transformations, train a classifier, and predict the disease class of
a single leaf image. Feature extraction relies on a leaf mask (PlantCV
thresholding), color statistics in RGB/HSV/LAB, and mask area. The classifier
is a scikit-learn `RandomForestClassifier`.

## 2. Requirements and setup

- Python 3.11 or newer. The pinned `numpy==2.4.6` and `pandas==3.0.6` require
  3.11+; the project was developed with Python 3.14.
- Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Direct dependencies, as found in the imports:

| Package | Used by |
| --- | --- |
| `opencv-python` (`cv2`) | all programs |
| `matplotlib` | Distribution, Augmentation, Transformation, predict |
| `numpy` | Augmentation, Transformation, train, predict |
| `plantcv` | Transformation (and train/predict through it) |
| `scikit-learn` | train, predict (RandomForest, StandardScaler, split, accuracy) |

The standard library covers the rest (`argparse`, `pickle`, `json`,
`zipfile`, `multiprocessing`, `shutil`, `re`). `requirements.txt` is a full
`pip freeze` and also contains transitive packages and `flake8`.

## 3. Project structure

```
.
├── Distribution.py       # Pie + bar chart of images per class
├── Augmentation.py       # 6 augmentations of one image; exports transform_registry
├── balance_dataset.py    # Balances every class by generating augmented images
├── clean_augmented.py    # Deletes augmented images (files ending in _<Transform>[N])
├── Transformation.py     # PlantCV transformations: display, batch, and helpers for the model
├── train.py              # Feature extraction, RandomForest training, writes learnings/ and learnings.zip
├── predict.py            # Predicts the class of one image from learnings/
├── requirements.txt      # Pinned dependencies
├── .gitignore
└── README.md
```

Git-ignored (see `.gitignore`):

| Path | Content |
| --- | --- |
| `images/`, `leaves/` | Dataset directories |
| `dst_directory/` | Output of batch `Transformation` |
| `learnings/`, `learnings.zip` | Trained model plus a copy of the dataset |
| `.venv/`, `venv/`, `env/`, `__pycache__/`, `.DS_Store`, `test.py` | Local environment and scratch files |

The subject forbids turning in the dataset, so every directory that holds
dataset images, including the zip that embeds them, must stay out of git.
`balance_dataset.py` writes to `augmented_directory/` next to the dataset
directory it was given. That path is only ignored if the dataset lives under
`images/` or `leaves/`.

## 4. Usage

Examples assume the dataset is at `images/Apple/`, with one subdirectory per
class (for example `apple_healthy/`, `apple_black_rot/`).

### Distribution

```bash
python3 Distribution.py images/Apple
```

Counts `.jpg`/`.jpeg`/`.png` files in each class subdirectory and opens a
matplotlib window with a pie chart (percentages) and a bar chart (counts).
Exits with `Error: ...` if the path is missing, is not a directory, has no
subdirectories, or no images.

### Augmentation (single image)

```bash
python3 Augmentation.py "images/Apple/apple_healthy/image (1).JPG"
```

Writes six files next to the source image and displays the original plus the
six results in one window:

```
image (1)_Flip.JPG
image (1)_Rotate.JPG
image (1)_Skew.JPG
image (1)_Shear.JPG
image (1)_Crop.JPG
image (1)_Distortion.JPG
```

Rotation angle is random in [-25, 25] degrees. Augmentation has no flags.

### Dataset balancing

Balancing is a separate script, not an option of `Augmentation.py`:

```bash
python3 balance_dataset.py images/Apple
```

Example output:

```
Building balanced dataset in images/augmented_directory/Apple
apple_black_rot: generating 1500 images (621 -> 2121)
apple_healthy: already balanced (2121)
```

Behavior:

- Copies the original images of every class into
  `<dataset parent>/augmented_directory/<dataset name>/<class>/`. The source
  dataset is not modified.
- The target size is the largest class's original count.
- Smaller classes receive augmented images until they reach the target.
  Output names follow `<stem>_<Transform>[N]<ext>` (`_Flip`, `_Flip2`, ...).
- The random seed is 42 (`balance_dataset(root, seed=42)`); the command line
  does not expose it.

To remove augmented images (any file whose stem ends in
`_Flip|_Rotate|_Skew|_Shear|_Crop|_Distortion` with an optional number),
recursively:

```bash
python3 clean_augmented.py images/augmented_directory/Apple
```

### Transformation

Display mode (one image, no files written):

```bash
python3 Transformation.py "images/Apple/apple_healthy/image (1).JPG"
```

Opens one window with Original, Gaussian blur, Mask, Roi objects, Analyze
object and Pseudolandmarks, then a second window with the color histogram.

Batch mode (every `.jpg`/`.jpeg`/`.png` directly inside `-src`, not recursive):

```bash
python3 Transformation.py -src images/Apple/apple_healthy -dst dst_directory
python3 Transformation.py -src images/Apple/apple_healthy -dst dst_directory -mask -blur
```

Flags: `-blur`, `-mask`, `-roi`, `-analyze`, `-landmarks`, `-hist`. With none
of them, all six outputs are written. Output files:

```
dst_directory/<name>_Gaussian_blur.<ext>
dst_directory/<name>_Mask.<ext>
dst_directory/<name>_Roi_objects.<ext>
dst_directory/<name>_Analyze_object.<ext>
dst_directory/<name>_Pseudolandmarks.<ext>
dst_directory/<name>_histogram.png
```

Console output: `Processed <file>` per image, `Skipped <file>: <reason>` on
failure, then `Done. Transformations saved in dst_directory`.

### train

```bash
python3 train.py images/augmented_directory/Apple
python3 train.py images/augmented_directory/Apple -out learnings -workers 4
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `directory` | required | Root with one subdirectory per class |
| `-out` | `learnings` | Output directory |
| `-workers` | `cpu_count() - 1` | Processes used for feature extraction |

Example output:

```
Found 8 classes: ['apple_black_rot', ...]
Found 16968 images total. Extracting features
Extracted features for 16968 images.
Train: 13575  Val: 3393
Validation accuracy: <value>%  (n=3393)
Saved model + dataset to learnings/, zipped as learnings.zip
```

(The numbers above are illustrative, not measured.) Details:

- Features (20 values): mean and standard deviation of each channel of the
  masked image in BGR, HSV and LAB, over foreground pixels, plus mask area and
  area ratio.
- Features are standardized (`StandardScaler`) and fed to a
  `RandomForestClassifier(n_estimators=300)`.
- Validation size is `max(20% of images, 100)`, stratified, `random_state=42`.
  The script exits with an error if that is not smaller than the dataset.
- Prints a warning if validation accuracy is below 90%, but still saves.
- `learnings/` contains `model.pkl`, `class_names.json`, `metrics.json` and a
  full copy of the dataset in `learnings/dataset/`. `learnings.zip` is built
  from that directory (relative to the current directory).

### predict

```bash
python3 predict.py "images/Apple/apple_healthy/image (1).JPG"
python3 predict.py "images/Apple/apple_healthy/image (1).JPG" -learnings learnings
```

Example output:

```
Class predicted : apple_healthy  (97.3%)
```

Also opens a window with the original and masked image. (Confidence value is
illustrative.)

## 5. Pipeline

Run from the repository root, in this order:

```bash
# 1. Inspect class imbalance
python3 Distribution.py images/Apple

# 2. (optional) Look at the augmentations of one image
#    Warning: this writes files into the dataset directory
python3 Augmentation.py "images/Apple/apple_healthy/image (1).JPG"

# 3. Build the balanced dataset in images/augmented_directory/Apple
python3 balance_dataset.py images/Apple

# 4. (optional) Check the balanced distribution
python3 Distribution.py images/augmented_directory/Apple

# 5. (optional) Look at the transformations
python3 Transformation.py "images/Apple/apple_healthy/image (1).JPG"
python3 Transformation.py -src images/Apple/apple_healthy -dst dst_directory

# 6. Train; produces learnings/ and learnings.zip
python3 train.py images/augmented_directory/Apple

# 7. Predict
python3 predict.py "images/Apple/apple_healthy/image (1).JPG"
```

If step 2 was run on an original image, remove the generated files before
step 3, otherwise they are treated as augmentations and excluded
(`clean_augmented.py images/Apple` removes them).

## 6. Team split and ownership

Ownership as confirmed by the team (two contributors). `git log` shows the balancing scripts committed under kaoutharrr.

| Part | Files | Author |
| --- | --- | --- |
| 1. Distribution | `Distribution.py` | belmqadem |
| 2. Augmentation | `Augmentation.py` (6 transforms and registry) | belmqadem |
| 3. Transformation | `Transformation.py` | kaoutharrr |
| 2/4. Balancing and cleaning | `balance_dataset.py`, `clean_augmented.py` | belmqadem |
| 4. Classification | `train.py`, `predict.py` | kaoutharrr |

## 7. Submission checklist

- [ ] Build `directory.zip` containing the dataset and `learnings/`
      (`train.py` already produces `learnings.zip` with both).
- [ ] Compute the hash:

```bash
sha1sum directory.zip
```

- [ ] Put the hash in `signature.txt` (one line, nothing else).
- [ ] Run `git status` and confirm that no dataset file (`images/`, `Apple/`,
      `augmented_directory/`, `learnings/`, any `.zip`) is staged or
      untracked-but-addable.
- [ ] Confirm the validation set has at least 100 images and accuracy above
      90%: check the `Train: ... Val: ...` and `Validation accuracy` lines of
      `train.py`, or `learnings/metrics.json`.

## 8. Design decisions

Only decisions the code implements are listed.

- **Originals are filtered during balancing (idempotency).**
  `balance_dataset.py` identifies augmented files by the regex
  `_(Flip|Rotate|Skew|Shear|Crop|Distortion)\d*$` on the file stem. Only
  non-matching files count as originals and are used as augmentation sources.
  The generation deficit is computed from the existing total and generated
  files are never overwritten, so re-running the script does not re-augment
  augmented images or keep growing the classes. The same regex is reused by
  `clean_augmented.py`.
- **Transforms are dispatched through a registry.** `transform_registry` in
  `Augmentation.py` maps names to functions. `Augmentation.py` iterates it for
  display, `balance_dataset.py` builds its (image, transform) combinations
  from it, and the file-name regex is generated from its keys, so adding a
  transform in one place updates all three.
- **The seed is a parameter of balancing.** `balance_dataset(root, seed=42)`
  seeds `random` so the choice and order of (image, transform) pairs is
  reproducible. It is not a command line option, and `Rotate` angles are only
  reproducible when going through that function. `train.py` uses fixed
  constants (`SEED = 42`) for the split and the forest.
- **Not implemented: splitting before augmentation.** `train.py` splits
  after balancing, on the already augmented directory, so augmented variants
  of one original can land in both the training and validation sets. The
  split is stratified but not grouped by source image.
