import os
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
from skimage import exposure, filters
import csv
import matplotlib.pyplot as plt
#os.chdir('Spatial-U-net for hand bone joint localization')
if __name__ == '__main__':

	train_path = './dataset/train/'
	valid_path = './dataset/valid/'
	test_path = './dataset/test/'
	GT_paths = './dataset/all_coordinates.csv' 

	for root in [train_path, valid_path, test_path]:

		image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

		for image_path in image_paths:
				filename = os.path.basename(image_path)[:-len(".jpg")]

				image = Image.open(image_path).convert('L')
				original_width, original_height = image.size
				image = F.resize(image, (256,256))

				# Convert to numpy array
				image_np = np.array(image)

				# Apply Otsu's thresholding to the input image
				otsu_threshold_input = filters.threshold_otsu(image_np)
				otsu_mask_input = image_np > otsu_threshold_input

				# Load and preprocess the reference image
				reference_image_path = './dataset/train/3209.jpg'
				reference_image = Image.open(reference_image_path).convert('L')
				reference_image_resized = F.resize(reference_image, (256, 256))
				reference_np = np.array(reference_image_resized)

				# Apply Otsu's thresholding to the reference image
				otsu_threshold_reference = filters.threshold_otsu(reference_np)
				otsu_mask_reference = reference_np > otsu_threshold_reference

				# Perform histogram matching within the masks
				matched = exposure.match_histograms(image_np[otsu_mask_input], reference_np[otsu_mask_reference], channel_axis=None)

				# Replace the segmented part of the input image with the matched result
				result = image_np.copy()
				result[otsu_mask_input] = matched

				# Convert back to PIL Image
				image = Image.fromarray(result.astype(np.uint8))
				
				matched_row = None
				scale_x = 256 / original_width
				scale_y = 256 / original_height

				with open(GT_paths, newline='') as csvfile:
					csvreader = csv.reader(csvfile)
					# Iterate through each row
					for row in csvreader:
						# Check if the first component matches the index
						if row[0] == filename:
							matched_row = row
							break

				if matched_row is not None:
					# Extract values except the first one
					coordinates = matched_row[1:]
				
					# Convert the extracted values to float32
					coordinates_values = np.array(coordinates, dtype=np.float32)
					# Split the coordinates into x and y
					x_coords = coordinates_values[0::2]*scale_x
					y_coords = coordinates_values[1::2]*scale_y
					print("Matched coordinate values:", x_coords, y_coords)
				else:
					print(f"No row found with the first component equal to {filename}")
				# Constants for the Gaussian function
				gamma = 100.0
				sigma = 2.0  # standard deviation
				d = 2  # Dimension
				width, height = image.size
				coordinates_values = np.column_stack((x_coords, y_coords))

				heatmap_images = []
				for i, (x0, y0) in enumerate(coordinates_values):
					x = np.linspace(0, width - 1, width)
					y = np.linspace(0, height - 1, height)
					x, y = np.meshgrid(x, y)
					# Generate the Gaussian heatmap
					heatmap = gamma/ ((2 * np.pi) ** (d / 2) * sigma**d)* np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
					# Normalize the heatmap to range [0, 1]
					heatmap = heatmap / np.max(heatmap)
					# Convert to image format
					heatmap_images.append(Image.fromarray((heatmap * 255).astype(np.uint8), mode="L"))

				# plot image and heatmap_images for checking picture variations
				root_input = './dataset/test_input/'
				if not os.path.exists(root_input):
					os.makedirs(root_input)
				image.save(os.path.join(root_input, f'{filename}.jpg'))

				root_GT = './dataset/test_GT/'
				if not os.path.exists(root_GT):
					os.makedirs(root_GT)
				for i, heatmap in enumerate(heatmap_images):
					heatmap.save(os.path.join(root_GT, f'{filename}_GT{i}.jpg'))
				
				# plot image and heatmaps together for checking
				if filename in ['3145','3149', '3190','3224']:
					summed_image = np.zeros_like(np.array(image), dtype=np.float32)

					for img in heatmap_images:
						img_array = np.array(img, dtype=np.float32)
						summed_image += img_array
					summed_image += np.array(image, dtype=np.float32)
					plt.imshow(summed_image, cmap='gray')
					plt.colorbar()
					plt.show()