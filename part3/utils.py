import matplotlib.pyplot as plt
import cv2

def display_images(images):
    """
    Display multiple images side by side.

    Parameters:
        images: list of tuples
            Example:
            [
                ("Original", original_image),
                ("Gaussian Blur", blurred_image),
                ("Mask", mask)
            ]
    """

    n = len(images)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    # If there's only one image, axes isn't a list
    if n == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, images):

        # Convert BGR -> RGB only if it's a color image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray")

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()