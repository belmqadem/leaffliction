# 🌿 Leaffliction

An innovative computer vision project that analyzes leaf images to extract visual features and support plant disease recognition.

---

# 📖 Table of Contents

- [What is Computer Vision?](#-what-is-computer-vision)
- [AI vs Machine Learning vs Computer Vision](#-ai-vs-machine-learning-vs-computer-vision)
- [OpenCV vs PlantCV](#-opencv-vs-plantcv)
- [Project Progress](#-project-progress)
- [Reading Images with PlantCV](#-reading-images-with-plantcv)
- [Gaussian Blur](#-gaussian-blur)

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

---

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

In this project, Gaussian Blur is implemented using PlantCV:

```python
blurred = pcv.gaussian_blur(
    img=image,
    ksize=(5, 5),
    sigma_x=0,
    sigma_y=None
)
```

The blurred image removes small variations while preserving the overall structure of the leaf, making later processing steps more reliable.

---

# 🔜 Next Steps

- [x] Read image
- [x] Gaussian Blur
- [ ] Mask Generation
- [ ] ROI Extraction
- [ ] Analyze Object
- [ ] Pseudolandmarks
- [ ] Color Histogram