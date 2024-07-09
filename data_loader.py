import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import functional as F
from PIL import Image

class ImageFolder(data.Dataset):
	def __init__(self, root,mode='train'):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		
		# GT : Ground Truth
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.mode = mode
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def deterministic_transform(self, image, rotation_degree, translate, scale):

		# Apply Rotation
		image = F.rotate(image, rotation_degree)
		
		# Apply Translation
		image = F.affine(image, angle=0, translate=translate, scale=scale, shear=0)
		
		return image

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		GT_paths = [image_path.replace('input', 'GT').replace('.jpg', f'_GT{i}.jpg') for i in range(37)]

		image = Image.open(image_path).convert('L')
		heatmap_images = []
		for path in GT_paths:
			heatmap_images.append(Image.open(path).convert('L'))
		
		if (self.mode == 'train'):
			#intensity augmentation
			image_array = np.array(image, dtype=np.float32)
			# Normalize the intensity values to the range [0, 1]
			normalized_image = image_array / 255.0
			# Scale the normalized values to the range [-1, 1]
			scaled_image = normalized_image * 2 - 1
			# Apply random intensity multiplication and shift
			intensity_multiplier = np.random.uniform(1, 1, size=scaled_image.shape)
			intensity_shift = np.random.uniform(-0.001, 0.001, size=scaled_image.shape)
			augmented_image = scaled_image * intensity_multiplier + intensity_shift
			min_val = augmented_image.min()
			max_val = augmented_image.max()
			augmented_image = 2 * (augmented_image - min_val) / (max_val - min_val) - 1
			# If needed, convert back to PIL Image for further processing
			image = Image.fromarray(((augmented_image + 1)/2 * 255).astype(np.uint8))
			
			# Example parameters for transformations
			rotation_degree = random.randint(-10, 10)
			translate_x = random.uniform(-20, 20)
			translate_y = random.uniform(-20, 20)
			scale = random.uniform(0.7, 1.2)

			# Apply the same transformations to both image and heatmap
			image = self.deterministic_transform(image, rotation_degree, (translate_x, translate_y), scale)
			heatmap_images = [self.deterministic_transform(heatmap, rotation_degree, (translate_x, translate_y), scale) for heatmap in heatmap_images]

		image_np = np.array(image, dtype=np.float32)
		image = (image_np / 255.0) * 2 - 1

		for i, heatmap in enumerate(heatmap_images):
			heatmap_np = np.array(heatmap, dtype=np.float32)
			heatmap_images[i] = (heatmap_np / 255.0) * 2 - 1

		# Convert the image and heatmaps to tensors
		image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
		heatmap_tensors = torch.tensor(np.stack(heatmap_images), dtype=torch.float32)  # Convert to tensor

		return image_tensor, heatmap_tensors

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, batch_size, num_workers=2, mode='train'):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, mode=mode)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader