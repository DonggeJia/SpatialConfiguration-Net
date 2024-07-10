import os
import sys
import torch
import torchvision
from torch import optim
from network import SCN
import csv
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output

class CustomLoss(nn.Module):
    def __init__(self, lambda_=0.00005):
        super(CustomLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, predicted_heatmaps, target_heatmaps, model_weights):
        # Heatmap loss (squared difference between predicted and target heatmaps)
        heatmap_loss = torch.sum((predicted_heatmaps - target_heatmaps) ** 2)
        # Regularization term for network weights (L2 norm)
        weights_loss = torch.sum(torch.stack([torch.sum(w ** 2) for w in model_weights if w.requires_grad]))

        # Combined loss
        loss = heatmap_loss + self.lambda_ * weights_loss
        return loss

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):
		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.SCN = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		self.SCN = SCN(img_ch=1, n_landmarks=37)

		self.optimizer = optim.Adam(list(self.SCN.parameters()), self.lr, [self.beta1, self.beta2])
		#self.optimizer = optim.SGD(list(self.SCN.parameters()), lr=1e-7, momentum=0.99, nesterov=True)

		self.SCN.to(self.device)

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.SCN.zero_grad()
		  
	def save_multichannel_image(self, tensor, base_path, base_name, epoch):
		num_channels = tensor.size(1)  # Get the number of channels
		for i in range(num_channels):
			channel_tensor = tensor[:, i, :, :].unsqueeze(1)  # Extract the i-th channel and add a channel dimension
			file_path = os.path.join(base_path, f'{base_name}_epoch_{epoch + 1}_channel_{i + 1}.png')
			torchvision.utils.save_image(channel_tensor, file_path)

	def getIPE(self, predicted_coordinates, GT_coordinates):
		distances = torch.norm(predicted_coordinates - GT_coordinates, dim=1)
		total_distance = torch.sum(distances) / predicted_coordinates.size(0)
		return total_distance

	def getNumber_outliers(self, predicted_coordinates, GT_coordinates, radius=10):
		"""
		Calculate the number of predicted landmarks that are outside a certain point-to-point error radius.
		"""
		distances = torch.norm(predicted_coordinates - GT_coordinates, dim=1)
		num_outliers = torch.sum(distances > radius).item()
		return num_outliers
	def get_coordinates_from_heatmap(self, heatmap):
		"""
		Extract the coordinates of the highest intensity points from the heatmap.
		heatmap: Tensor of shape (1, 37, 256, 256)
		Returns: Tensor of shape (37, 2) containing the coordinates
		"""
		coords = []
		for i in range(heatmap.shape[1]):
			channel = heatmap[0, i, :, :]
			max_pos = torch.argmax(channel).item()
			coord = torch.tensor([max_pos % 256, max_pos // 256], dtype=torch.float32)
			coords.append(coord)
		return torch.stack(coords)
	def train(self):
		"""Train encoder, generator, and discriminator."""
		fig, ax = plt.subplots(2, 1, figsize=(10, 8))
		epoch_loss = []
		
		SCN_path = os.path.join(self.model_path, '%s-%d-%.4f-%d.pkl' % (self.model_type, self.num_epochs, self.lr, self.num_epochs_decay))

		# SCN Train
		if os.path.isfile(SCN_path):
			self.SCN.load_state_dict(torch.load(SCN_path))
			print('%s is Successfully Loaded from %s' % (self.model_type, SCN_path))
		else:
			lr = self.lr
			criterion = CustomLoss().to(self.device)
			best_IPE_ = 100000.
			for epoch in range(self.num_epochs):
				self.SCN.train(True)
				epoch_loss = 0
				IPE_ = 0.0
				Number_outliers_ = 0.0
				length = 0
				for i, (images, GT) in enumerate(self.train_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					predict_heatmap = self.SCN(images)

					loss = criterion(predict_heatmap, GT, self.SCN.parameters())
					epoch_loss += loss.item()
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					predicted_coords = self.get_coordinates_from_heatmap(predict_heatmap)
					GT_coords =self.get_coordinates_from_heatmap(GT)

					IPE_ += self.getIPE(predicted_coords, GT_coords)
					Number_outliers_ += self.getNumber_outliers(predicted_coords, GT_coords, radius=4) #radius means the distance in number of pixels                   
					
					length += images.size(0)

				IPE_ = IPE_ / length
				Number_outliers_ = Number_outliers_ / length

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n--[Training] IPE: %.4f, Or: %.4f' % (
					epoch + 1, self.num_epochs, epoch_loss, IPE_, Number_outliers_))
				sys.stdout.flush()
				
				clear_output(wait=True)
				ax[0].plot(epoch +1, epoch_loss, 'o', label='Train Loss')
				ax[0].set_xlabel('Epoch')
				ax[0].set_ylabel('Loss')
			
				ax[1].plot(epoch +1, IPE_, 'o', label='Train IPE')
				ax[1].set_xlabel('Epoch')
				ax[1].set_ylabel('IPE')
			
				plt.tight_layout()
				plt.pause(0.1)


				
				# Decay learning rate
				if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print('----Decay learning rate to lr: {}.'.format(lr))

				#===================================== Validation ====================================#
				self.SCN.train(False)
				self.SCN.eval()

				IPE_ = 0.0
				Number_outliers_ = 0.0
				length = 0

				for i, (images, GT) in enumerate(self.valid_loader):
					images = images.to(self.device)
					GT = GT.to(self.device)
					# Predicted heatmap
					predict_heatmap = self.SCN(images)				
					predicted_coords = self.get_coordinates_from_heatmap(predict_heatmap)
					GT_coords =self.get_coordinates_from_heatmap(GT)

					IPE_ += self.getIPE(predicted_coords, GT_coords)
					Number_outliers_ += self.getNumber_outliers(predicted_coords, GT_coords, radius=4)  					
					length += images.size(0)
				IPE_ = IPE_ / length # average IPE for this batch
				Number_outliers_ = Number_outliers_ / length

				# Print the log info
				print('----[Validation] IPE: %.4f, Or: %.4f' %(IPE_, Number_outliers_))
				sys.stdout.flush()
				self.save_multichannel_image(images.data.cpu(), self.result_path, f'{self.model_type}_valid_image', epoch)
				self.save_multichannel_image(predict_heatmap.data.cpu(), self.result_path, f'{self.model_type}_valid_heatmap', epoch)
				self.save_multichannel_image(GT.data.cpu(), self.result_path, f'{self.model_type}_valid_GT', epoch)

				# Save Best U-Net model
				if IPE_ < best_IPE_:
					best_IPE_ = IPE_
					best_epoch = epoch
					best_SCN = self.SCN.state_dict()
					print('----Best %s model IPE : %.4f'%(self.model_type, best_IPE_))
					torch.save(best_SCN,SCN_path)
					
			#===================================== Test ====================================#
			del self.SCN
			del best_SCN
			self.build_model()
			self.SCN.load_state_dict(torch.load(SCN_path))
			
			self.SCN.train(False)
			self.SCN.eval()

			IPE_ = 0.0
			Number_outliers_ = 0.0
			length = 0

			for i, (images, GT) in enumerate(self.test_loader):
				images = images.to(self.device)
				GT = GT.to(self.device)
				# Predicted heatmap
				predict_heatmap = self.SCN(images)				
				predicted_coords = self.get_coordinates_from_heatmap(predict_heatmap)
				GT_coords =self.get_coordinates_from_heatmap(GT)

				IPE_ += self.getIPE(predicted_coords, GT_coords)
				Number_outliers_ += self.getNumber_outliers(predicted_coords, GT_coords, radius=4)  				
				length += images.size(0)
			IPE_ = IPE_ / length # average IPE for this batch
			Number_outliers_ = Number_outliers_ / length



			f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow([self.model_type,IPE_,Number_outliers_,self.lr, best_epoch,self.num_epochs,self.num_epochs_decay])
			f.close()
