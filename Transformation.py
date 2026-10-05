#!/usr/bin/env python3

from pathlib import Path
import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

pcv.params.debug = None


def read_image(path: str):
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if not img_path.is_file():
        raise IsADirectoryError(f"{path} is a directory, not a file.")

    img, _, _ = pcv.readimage(filename=path)
    return img


def gaussian_blur(img, ksize=5):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel="l")
    blur = pcv.gaussian_blur(img=gray, ksize=(ksize, ksize), sigma_x=0)
    return blur


def build_mask(img, threshold=115, fill_size=200, min_foreground_px=50):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel="l")
    thresh = pcv.threshold.binary(
        gray_img=gray, threshold=threshold, object_type="dark"
    )
    inv = pcv.invert(gray_img=thresh)
    if np.sum(thresh > 0) < np.sum(inv > 0) and np.sum(thresh > 0) < (
        0.5 * thresh.size
    ):
        mask = thresh
    else:
        mask = thresh if np.sum(thresh > 0) < np.sum(inv > 0) else inv

    if np.sum(mask > 0) < min_foreground_px and gray.std() > 1e-6:
        try:
            otsu = pcv.threshold.otsu(gray_img=gray, object_type="dark")
            otsu_inv = pcv.invert(gray_img=otsu)
            mask = (
                otsu
                if np.sum(otsu > 0) < np.sum(otsu_inv > 0)
                else otsu_inv
            )
        except Exception:
            pass

    mask = pcv.fill(bin_img=mask, size=fill_size)
    mask = pcv.dilate(gray_img=mask, ksize=2, i=1)
    mask = pcv.erode(gray_img=mask, ksize=2, i=1)

    if np.sum(mask > 0) < min_foreground_px:
        mask = np.full(gray.shape, 255, dtype=np.uint8)

    return mask


def apply_mask(img, mask):
    return pcv.apply_mask(img=img, mask=mask, mask_color="white")


def roi_objects(img, mask):
    h, w = mask.shape[:2]
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=h, w=w)
    kept_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")

    vis = img.copy()
    vis[kept_mask > 0] = (0, 255, 0)
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (255, 0, 0), 4)
    return vis, kept_mask


def analyze_object(img, mask):
    shape_img = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1)
    return shape_img


def pseudolandmarks(img, mask):
    vis = img.copy()
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=vis, mask=mask, label="default"
    )

    def _draw(points, color):
        for p in points:
            x, y = int(p[0][0]), int(p[0][1])
            cv2.circle(vis, (x, y), 3, color, -1)

    _draw(top, (255, 0, 255))
    _draw(bottom, (255, 0, 0))
    _draw(center_v, (0, 128, 255))
    return vis


def color_histogram_figure(img, mask, save_path=None):
    try:
        _ = pcv.analyze.color(
            rgb_img=img, labeled_mask=mask, colorspaces="all"
        )
    except Exception as exc:
        raise RuntimeError(f"color analysis failed: {exc}") from exc
    hist_data = pcv.outputs.observations["default_1"]

    fig, ax = plt.subplots(figsize=(7, 5))
    channel_map = {
        "blue_frequencies": ("blue", "blue"),
        "green_frequencies": ("green", "green"),
        "red_frequencies": ("red", "red"),
        "lightness_frequencies": ("lightness", "gray"),
        "green-magenta_frequencies": ("green-magenta", "magenta"),
        "blue-yellow_frequencies": ("blue-yellow", "gold"),
        "hue_frequencies": ("hue", "purple"),
        "saturation_frequencies": ("saturation", "cyan"),
        "value_frequencies": ("value", "orange"),
    }
    for key, (label, color) in channel_map.items():
        if key in hist_data:
            values = hist_data[key]["value"]
            total = sum(values) if sum(values) > 0 else 1
            pct = [100.0 * v / total for v in values]
            ax.plot(pct, label=label, color=color, linewidth=1)

    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.set_title("Color histogram")
    ax.legend(title="color Channel", loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    return fig


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


def transform_for_model(img):
    mask = build_mask(img)
    masked = apply_mask(img, mask)
    return cv2.resize(masked, (128, 128))


def show_single_image(path):
    img = read_image(path)
    results = compute_all(img)
    mask_for_hist = results.pop("_mask_for_hist")

    names = list(results.keys())
    n = len(names)

    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows)
    )

    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, names):
        im = results[name]

        im_rgb = (
            cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if im.ndim == 3
            else im
        )

        cmap = None if im.ndim == 3 else "gray"

        ax.imshow(im_rgb, cmap=cmap)
        ax.set_title(name.replace("_", " "))
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()

    # DISPLAY the transformations
    plt.show()

    # Display the color histogram
    try:
        color_histogram_figure(
            img,
            mask_for_hist
        )
        plt.show()
    except RuntimeError as exc:
        print(f"Color histogram skipped: {exc}")


def save_batch(src_dir, dst_dir, flags):
    os.makedirs(dst_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    files = [f for f in os.listdir(src_dir) if f.endswith(exts)]

    if not files:
        print(f"No images found in {src_dir}")
        return

    do_all = not any(flags.values())

    skipped = []
    for fname in files:
        path = os.path.join(src_dir, fname)
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
                    out_path = os.path.join(
                        dst_dir, f"{base}_{result_key}{ext}"
                    )
                    cv2.imwrite(out_path, out_img)

            if do_all or flags.get("hist"):
                try:
                    hist_path = os.path.join(dst_dir, f"{base}_histogram.png")
                    fig = color_histogram_figure(
                        img, mask_for_hist, save_path=hist_path
                    )
                    plt.close(fig)
                except RuntimeError as hist_exc:
                    print(f"  (histogram skipped for {fname}: {hist_exc})")

            print(f"Processed {fname}")
        except Exception as exc:
            skipped.append(fname)
            print(f"Skipped {fname}: {exc}")

    print(f"Done. Transformations saved in {dst_dir}")
    if skipped:
        print(f"{len(skipped)} image(s) skipped due to errors: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Leaffliction — image transformation tool (Part 3)."
    )
    parser.add_argument(
        "image", nargs="?", help="Path to a single image (display mode)."
    )
    parser.add_argument(
        "-src", dest="src", help="Source directory (batch mode)."
    )
    parser.add_argument(
        "-dst", dest="dst", help="Destination directory (batch mode)."
    )
    parser.add_argument(
        "-blur", action="store_true", help="Include Gaussian blur."
    )
    parser.add_argument("-mask", action="store_true", help="Include mask.")
    parser.add_argument(
        "-roi", action="store_true", help="Include ROI objects."
    )
    parser.add_argument(
        "-analyze", action="store_true", help="Include analyze object."
    )
    parser.add_argument(
        "-landmarks", action="store_true", help="Include pseudolandmarks."
    )
    parser.add_argument(
        "-hist", action="store_true", help="Include color histogram."
    )

    args = parser.parse_args()

    try:
        if args.src and args.dst:
            flags = {
                "blur": args.blur,
                "mask": args.mask,
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
    except (FileNotFoundError, IsADirectoryError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
