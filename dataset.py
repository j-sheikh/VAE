import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class RGBDDataset(Dataset):
	def __init__(self, folder_path, transform=None):
		self.folder_path = folder_path
		self.episode_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
		self.transform = transform

	def __len__(self):
		return len(self.episode_files)

	def __getitem__(self, idx):
		episode_path = os.path.join(self.folder_path, self.episode_files[idx])
		episode = np.load(episode_path)

		# Load and preprocess images from the episode
		image = episode['depth_gripper']
		image = image / 2.0

		return image

class RGBDDataModule(pl.LightningDataModule):
	def __init__(self, folder_path, batch_size, num_episodes=30, num_workers = 2):
		super().__init__()

		self.num_workers = num_workers
		self.folder_path = folder_path
		self.batch_size = batch_size
		self.num_episodes = num_episodes
		self.data_transform = transforms.Compose([
			transforms.ToTensor()
		])



	def setup(self, stage=None):

		self.dataset = RGBDDataset(self.folder_path, transform= self.data_transform)
		num_samples = len(self.dataset)

		num_selected_episodes = int(self.num_episodes / 100 * num_samples)
		selected_indices = np.random.choice(num_samples, num_selected_episodes, replace=False)
		remaining_indices = np.setdiff1d(np.arange(num_samples), selected_indices)

		self.selected_dataset = torch.utils.data.Subset(self.dataset, selected_indices)
		self.remaining_dataset = torch.utils.data.Subset(self.dataset, remaining_indices)

		num_remaining_samples = len(self.remaining_dataset)
		num_val_samples = int(0.1 * num_remaining_samples)
		num_test_samples = int(0.1 * num_remaining_samples)

		self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
			self.remaining_dataset,
			[num_remaining_samples - num_val_samples - num_test_samples, num_val_samples, num_test_samples]
		)


	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


"""
batch_size = 64
num_episodes = 10
num_workers = 8

folder_path =  '/media/jannik/media/thesis/GitHub/hulc/dataset/task_ABC_D/training'
data_module = RGBDDataModule(folder_path=folder_path, batch_size=batch_size, num_episodes=num_episodes, num_workers=num_workers)
data_module.setup()
train_dataloader = data_module.train_dataloader()
num_iterations = 20

for _ in range(num_iterations):
    batch = next(iter(train_dataloader))
    x = batch
    min_value = x.min().item()
    max_value = x.max().item()
    print("Minimum value in batch:", min_value)
    print("Maximum value in batch:", max_value)

batch = next(iter(train_dataloader))
x = batch
grid = make_grid(x.unsqueeze(1), nrow=8)
grid_np = np.transpose(grid.numpy(), (1, 2, 0))
plt.imshow(grid_np )
plt.axis('off')
plt.show()


episode_files = [f for f in os.listdir(root_dir) if f.endswith('.npz')]
episode_path = os.path.join(root_dir, episode_files[0])
episode = np.load(episode_path)
depth_gripper = episode['depth_gripper']
rgb_gripper = episode['rgb_gripper']
combined_image = np.concatenate((rgb_gripper, depth_gripper[..., None]), axis=-1)
plt.imshow(combined_image)
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(depth_gripper, cmap='gray')
plt.title('Depth Image')
plt.axis('off')
plt.show()

# Plot the RGB image
plt.figure(figsize=(6, 6))
plt.imshow(rgb_gripper)
plt.title('RGB Image')
plt.axis('off')
plt.show()
"""