import argparse
import csv
import glob
import os

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from skimage import exposure, filters
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_IMAGE_PATH = "train/3209.jpg"  # Picked as it has higher contrast
GAUSSIAN_GAMMA = 100.0
GAUSSIAN_SIGMA = 2.0


def load_image(
    image_path: str, target_width: int = 256, target_height: int = 256
) -> Tuple[np.ndarray, float, float]:
    """Load an image from the given path and resize it to the target dimensions.

    Args:
        image_path (str): the path to the image file.
        target_width (int, optional): the target width. Defaults to 256.
        target_height (int, optional): the target height. Defaults to 256.

    Returns:
        Image.Image: the loaded and resized image as a NumPy array.
        float: the width scaling factor.
        float: the height scaling factor.
    """
    image = Image.open(image_path).convert("L")
    original_width, original_height = image.size
    image = F.resize(image, (target_width, target_height))

    return (
        np.array(image),
        target_width / original_width,
        target_height / original_height,
    )


def get_otsu_mask(image_np: np.ndarray) -> np.ndarray:
    """
    Get the mask using Otsu's method for the input image.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The mask from Otsu's method.

    """
    otsu_threshold = filters.threshold_otsu(image_np)
    otsu_mask = image_np > otsu_threshold
    return otsu_mask


def apply_gaussian_heatmap(
    image: Image.Image,
    coordinates: np.ndarray,
    gamma: float = 100.0,
    sigma: float = 2.0,
) -> List[Image.Image]:
    """
    Apply a Gaussian heatmap to the input image at the given coordinates.

    Parameters:
        image (Image.Image): The input image.
        coordinates (np.ndarray): The coordinates to apply the heatmap with shape (N, 2).
        gamma (float, optional): The gamma value. Defaults to 100.0.
        sigma (float, optional): The sigma value. Defaults to 2.0.

    Returns:
        Image.Image: The image with the Gaussian heatmap applied.
    """
    assert len(image.size) == 2, ValueError("The input image must be grayscale")
    assert coordinates.shape[1] == 2, ValueError(
        "The coordinates must have shape (N, 2)"
    )

    d = 2  # Dimension
    width, height = image.size

    heatmap_images = []
    for x0, y0 in coordinates:
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        x, y = np.meshgrid(x, y)
        # Generate the Gaussian heatmap
        heatmap = (
            gamma
            / ((2 * np.pi) ** (d / 2) * (sigma**d))
            * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma**2)))
        )
        # Normalize the heatmap to range [0, 1]
        heatmap = heatmap / np.max(heatmap)
        # Convert to image format
        heatmap_images.append(
            Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
        )

    return heatmap_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, default="./dataset")
    parser.add_argument("--gaussian-gamma", type=float, default=100.0)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    args = parser.parse_args()

    input_paths, GT_output_paths, image_output_paths = [], [], []
    for root in ["train", "valid", "test"]:
        input_paths.append(os.path.join(args.root_path, root))
        GT_output_paths.append(os.path.join(args.root_path, f"{root}_GT"))
        image_output_paths.append(os.path.join(args.root_path, f"{root}_input"))

    GT_path = os.path.join(args.root_path, "all_coordinates.csv")

    reference_np, _, _ = load_image(os.path.join(args.root_path, REFERENCE_IMAGE_PATH))
    reference_otsu_mask = get_otsu_mask(reference_np)

    GT_mapping = {}
    with open(GT_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            GT_mapping[row[0]] = row[1:]

    for input_path, GT_output_path, image_output_path in zip(
        input_paths, GT_output_paths, image_output_paths
    ):
        image_paths = glob.glob(os.path.join(input_path, "*"))

        for image_path in image_paths:
            stem = Path(image_path).stem
            image_np, scale_width, scale_height = load_image(image_path)
            image_otsu_mask = get_otsu_mask(image_np)

            # Perform histogram matching within the masks
            matched = exposure.match_histograms(
                image_np[image_otsu_mask],
                reference_np[reference_otsu_mask],
                channel_axis=None,
            )
            # Replace the segmented part of the input image with the matched result
            result = image_np.copy()
            result[image_otsu_mask] = matched
            # Convert back to PIL Image
            image = Image.fromarray(result.astype(np.uint8))

            try:
                coordinates_values = np.array(GT_mapping[stem], dtype=np.float32)
                x_coords = coordinates_values[0::2] * scale_width
                y_coords = coordinates_values[1::2] * scale_height
                coordinates = np.column_stack((x_coords, y_coords))
                print("Matched coordinate values:", x_coords, y_coords)
            except KeyError:
                print(
                    f"No row found with the first component equal to {stem} ({image_path})"
                )

            heatmap_images = apply_gaussian_heatmap(
                image, coordinates, args.gaussian_gamma, args.gaussian_sigma
            )

            if not os.path.exists(image_output_path):
                os.makedirs(image_output_path)
            image.save(os.path.join(image_output_path, f"{stem}.jpg"))

            if not os.path.exists(GT_output_path):
                os.makedirs(GT_output_path)
            for i, heatmap in enumerate(heatmap_images):
                heatmap.save(os.path.join(GT_output_path, f"{stem}_GT{i}.jpg"))

            # plot image and heatmaps together for checking
            if stem in ["3145", "3149", "3190", "3224"]:
                summed_image = np.zeros_like(np.array(image), dtype=np.float32)

                for img in heatmap_images:
                    img_array = np.array(img, dtype=np.float32)
                    summed_image += img_array
                summed_image += np.array(image, dtype=np.float32)
                plt.imshow(summed_image, cmap="gray")
                plt.colorbar()
                plt.show()
