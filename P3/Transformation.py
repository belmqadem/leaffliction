import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

pcv.params.debug = None



def read_image(path):
    img, _, _= pcv.readimage(filename=path)
    return img

def gaussian_blur(img, ksize=5):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel="l")
    blur = pcv.gaussian_blur(img=gray, ksize=(ksize, ksize), sigma_x=0)
    return blur

def build_mask(img, threshold=115, full_size=200, min_foreground_px=50):
    
def compute_all(img):
    results = {}
    results["Original"] = img

    blur = gaussian_blur(img)
    results["Gaussian_blur"] = blur

    mask = build_mask(img)
    results["Mask"] = apply_mask(img, mask)

    roi_img, kept_mask = roi_objects(img, mask)
    results["Roi_objects"] = roi_img

    shape_img = analyze_object(img, kept_mask)
    results["Analyze_object"] = shape_img

    landmarks_img = pseudolandmarks(img, kept_mask)
    results["Pseudolandmarks"] = landmarks_img

    results["_mask_for_hist"] = kept_mask
    return results
def color_histogram_figure():
    
def save_batch(src, dst, flags):
    os.makedirs(dst, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files= [f for f in os.listdir(src) if f.endswith(exts)]
    if not files:
        print(f"No imgs found in {src}")
        return
    do_all = not any(flags.values())
    skipped=[] 
    for fname in files:
        path= os.path.join(src, fname)
        base, ext = os.path.splitext(fname)
        try:
            img = read_image(path)
            results = compute_all(img)
            mask_for_hist = results.pop("_mask_for_hist")

            wanted = {
                "blur": "Gaussian_blur",
                "mask": "Mask",
                "roi": "Roi_objects",
                "analyze": "Analyze_object",
                "landmarks": "Pseudolandmarks",
            }
            for flag_name, result_key in wanted.items():
                if do_all or flags.get(flag_name):
                    out_img = results[result_key]
                    out_path= os.path.join(dst, f"{base}_{result_key}{ext}")
                    cv2.imwrite(out_path,out_img)
            if do_all or flags.get("hist"):
                try:
                    hist_path = os.path.join(dst, f"{base}_histogram.png")
                    fig = color_histogram_figure(img, mask_for_hist,save_path=hist_path)
                    plt.close(fig)
                except RuntimeError as hist_exc:
                    print(f"  (histogram skipped for {fname}: {hist_exc})")
            print(f"Processed{fname}")
        except Exception as exc:
            skipped.append(fname)
            print(f"Skipped {fname}: {exc}")
    print(f"Done ,Transformation saved in {dst}")
    if skipped:
        print(f"{len (skipped)} imgs skipped due to errors")
  


def show_single_image(path):
    img = read_image(path)
    results = compute_all(img)
    mask_for_hist = results.pop("_mask_for_hist")
    names = list(results.keys())
    n = len(names)
    cols = 3
    rows = (n + cols -1) // cols
    fig, axes = plt.subplot(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax , name in zip(axes, names):
        im = results[name]
        img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im
        cmap= NONE if im.ndim ==3 else "gray"
        ax.imshow(img_rgb, cmap=cmap)
        ax.set_title(name.replace("_", " "))
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    grid_path = os.path.join( out_dir, "Transformation_grid.png")
    fig.savefig(grid_path, dpi=120)
    print(f"Saved transformation grid to {grid_path}")
    hist_path = os.path.join(out_dir, "transformation_histogram.png")
    try:
        color_histogram_figure(img, mask_for_hist, save_path=hist_path)
        print(f"Saved color histogram to {hist_path}")
    except RuntimeError as exc:
        print(f"Color histogram skipped: {exc}")

    

def main():

    parser = argparse.ArgumentParser(
        description="image transformation"
    )
    parser.add_argument(
        "image", nargs="?", help="Path to a single image (display mode)."
    )
    parser.add_argument("-src", dest="src", help="Source directory (batch mode).")
    parser.add_argument("-dst", dest="dst", help="Destination directory (batch mode).")
    parser.add_argument("-blur", action="store_true", help="Include Gaussian blur.")
    parser.add_argument("-mask", action="store_true", help="Include mask.")
    parser.add_argument("-roi", action="store_true", help="Include ROI objects.")
    parser.add_argument(
        "-analyze", action="store_true", help="Include analyze object."
    )
    parser.add_argument(
        "-landmarks", action="store_true", help="Include pseudolandmarks."
    )
    parser.add_argument("-hist", action="store_true", help="Include color histogram.")

    args = parser.parse_args()

    if args.src and args.dst:
        flags= {
            "blur": args.blur,
            "mask":args.mask,
            "roi": args.roi,
            "analyze": args.analyze,
            "landmarks": args.landmarks,
            "hist": args.hist,
        }
        save_batch(args.src, args.dst, flags)
    elif args.image:
        show_single_image(args.image)
    else:
        parser.print_help()
        sys.exit(1)





if __name__ == "__main__":
    main()
