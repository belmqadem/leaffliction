from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from utils import display_images
import sys

if len(sys.argv) != 2:
    print("Usage: python Transformation.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
image, path, filename=pcv.readimage(image_path)
blurred = pcv.gaussian_blur(img=image, ksize=(5,5), sigma_x=0, sigma_y=None)
display_images([
    ("Original", image),
    ("Gaussian Blur", blurred),
])
print(image)
