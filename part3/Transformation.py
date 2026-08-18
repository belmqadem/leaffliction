
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
# matplotlib.use("Agg") 
import matplotlib.pyplot as plt  

from plantcv import plantcv as pcv  

def read_image(path):
    img,_,_=pcv.readimage(path)
    return img
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="img path")
    args = parser.parse_args()
    print(read_image(args.image))

if __name__ == "__main__":
    main()