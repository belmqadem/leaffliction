# 🌿 Leaffliction — Part 3: Image Transformation

A computer vision pipeline that extracts visual features from leaf images to support plant disease recognition, built with PlantCV.

---

# 📖 Table of Contents

- [What is Computer Vision?](#-what-is-computer-vision)
- [AI vs Machine Learning vs Computer Vision](#-ai-vs-machine-learning-vs-computer-vision)
- [OpenCV vs PlantCV](#-opencv-vs-plantcv)
- [Project Progress](#-project-progress)
- [Reading Images with PlantCV](#-reading-images-with-plantcv)
- [Gaussian Blur](#-gaussian-blur)
- [Mask Generation](#-mask-generation)
- [ROI Objects](#-roi-objects)
- [Analyze Object](#-analyze-object)
- [Pseudolandmarks](#-pseudolandmarks)
- [Color Histogram](#-color-histogram)
- [Usage](#-usage)

---

# 🖥️ What is Computer Vision?

Computer Vision (CV) is a field of Artificial Intelligence that enables computers to understand and analyze images or videos.

Instead of humans looking at a picture, the computer receives an image represented as a NumPy array of pixels. Computer vision algorithms process these pixels to extract meaningful information such as colors, shapes, textures, edges, and objects.

In this project, Computer Vision is used to analyze leaf images before extracting features that can later help distinguish healthy leaves from diseased ones.

---

# 🤖 AI vs Machine Learning vs Computer Vision

Artificial Intelligence (AI)
- The broad field of creating systems capable of performing tasks that normally require human intelligence.

Machine Learning (ML)
- A subset of AI where computers learn patterns from data instead of being explicitly programmed.

Computer Vision (CV)
- A branch of AI focused on understanding images and videos.

Relationship:

```
Artificial Intelligence
│
├── Machine Learning
│
├── Natural Language Processing
│
└── Computer Vision
```

Machine Learning can be used inside Computer Vision, but many Computer Vision techniques (such as Gaussian Blur, thresholding, morphology, etc.) are classical image processing algorithms and do not require machine learning.

---

# 🌱 OpenCV vs PlantCV

## OpenCV

OpenCV is a general-purpose Computer Vision library.

It contains hundreds of image processing algorithms including:

- Image filtering
- Edge detection
- Feature detection
- Object detection
- Video processing
- Camera calibration
- Image transformations

It is designed for many Computer Vision applications beyond plant analysis.

## PlantCV

PlantCV is built on top of OpenCV.

It provides high-level functions specifically designed for plant phenotyping and leaf analysis.

Instead of implementing many image processing steps manually, PlantCV offers ready-to-use functions for common plant analysis tasks such as:

- Reading images
- Image segmentation
- ROI extraction
- Object analysis
- Color histograms
- Landmark extraction

For the Leaffliction project, PlantCV significantly simplifies the implementation of plant-specific image processing pipelines.

---

# 🚀 Project Progress

## ✅ Image Loading

The first step of any Computer Vision pipeline is loading the image.

PlantCV provides the `readimage()` function:

```python
image, path, filename = pcv.readimage(image_path)
```

This function returns:

- **image** → NumPy array containing the image pixels.
- **path** → directory containing the image.
- **filename** → image filename.

The loaded image is represented internally as a three-dimensional NumPy array:

```
(height, width, channels)
```

Example:

```
(256, 256, 3)
```

where:

- Height → number of rows.
- Width → number of columns.
- Channels → RGB color values.

Each pixel contains three values representing its color.

Example:

```
[119, 118, 122]
```

---

# 🌫️ Gaussian Blur

Gaussian Blur is an image smoothing technique used to reduce noise and small image details before further processing.

Instead of replacing each pixel with its original value, the algorithm computes a weighted average of neighboring pixels using a Gaussian kernel.

Pixels closer to the center have greater influence than pixels farther away.

Example of a 5×5 Gaussian kernel:

```
1   4   6   4   1
4  16  24  16   4
6  24  36  24   6
4  16  24  16   4
1   4   6   4   1
```

Implemented with PlantCV:

```python
gray = pcv.rgb2gray_lab(rgb_img=image, channel="l")
blurred = pcv.gaussian_blur(
    img=gray,
    ksize=(5, 5),
    sigma_x=0
)
```

The blurred image removes small variations while preserving the overall structure of the leaf, making later processing steps more reliable.

---

# 🎭 Mask Generation

Once the image is smoothed, the leaf needs to be separated from the background. This is done through **binary thresholding**: every pixel above (or below) a chosen intensity becomes white (foreground), everything else becomes black (background).

```python
gray = pcv.rgb2gray_lab(rgb_img=image, channel="l")
thresh = pcv.threshold.binary(
    gray_img=gray,
    threshold=115,
    object_type="dark"
)
mask = pcv.fill(bin_img=thresh, size=200)
```

- `pcv.threshold.binary` converts the grayscale image into a black/white mask.
- `pcv.fill` removes small noisy specks below a given pixel-area threshold, so only the actual leaf shape survives.

The result is a clean binary mask isolating the leaf from its background, which every following step depends on.

---

# 🟩 ROI Objects

A **Region Of Interest (ROI)** defines the area of the image we actually care about. PlantCV compares the mask against the ROI and keeps only the objects that fall inside it, discarding background noise outside the region.

```python
roi = pcv.roi.rectangle(img=image, x=0, y=0, h=height, w=width)
kept_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
```

This step guarantees that only the leaf (and not stray artifacts elsewhere in the frame) is passed on to the analysis steps.

---

# 📐 Analyze Object

`pcv.analyze.size` computes shape statistics on the segmented leaf — area, perimeter, convex hull, height/width, and more — and draws an overlay (outline + centroid + bounding lines) directly on the image.

```python
shape_image = pcv.analyze.size(
    img=image,
    labeled_mask=kept_mask,
    n_labels=1
)
```

These measurements are the kind of quantitative shape features that can later distinguish a diseased leaf (irregular, spotted edges) from a healthy one.

---

# 📍 Pseudolandmarks

Pseudolandmarks are evenly-spaced reference points placed along the leaf's top, bottom, and center vertical axis. They give a consistent, comparable set of coordinates across every leaf image regardless of size or orientation.

```python
top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
    img=image,
    mask=kept_mask,
    label="default"
)
```

These points are useful for shape comparison and alignment between leaves of the same or different classes.

---

# 🎨 Color Histogram

The color histogram shows how pixel intensities are distributed across multiple color spaces (RGB, HSV, LAB) for the segmented leaf.

```python
pcv.analyze.color(
    rgb_img=image,
    labeled_mask=kept_mask,
    colorspaces="all"
)
```

Each channel — red, green, blue, hue, saturation, value, lightness, green-magenta, blue-yellow — is plotted as a percentage of pixels at each intensity value (0–255). Differences in these distributions (e.g. a spike in brown/yellow tones) are strong signals for disease detection.

---

# ✅ Project Progress

- [x] Read image
- [x] Gaussian Blur
- [x] Mask Generation
- [x] ROI Extraction
- [x] Analyze Object
- [x] Pseudolandmarks
- [x] Color Histogram

---

# 🛠️ Usage

Single image (displays all transformations + saves the grid and histogram next to the image):

```bash
python3 main.py path/to/image.JPG
```

Batch mode (processes every image in a source directory, saves outputs into a destination directory):

```bash
python3 Transformation.py -src Apple/apple_healthy/ -dst dst_directory
python3 Transformation.py -src Apple/apple_healthy/ -dst dst_directory -mask -blur
```

If no transformation flags are passed in batch mode, all transformations are generated by default.