import torch
from torch import nn


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super().__init__()
		self.conv1 = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=3,
			stride=stride,
			padding=1,
			bias=False,
		)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(
			out_channels,
			out_channels,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=False,
		)
		self.bn2 = nn.BatchNorm2d(out_channels)

        # skip connection projection if needed to match dimensions
		if stride != 1 or in_channels != out_channels:
			self.proj = nn.Sequential(
				nn.Conv2d(
					in_channels,
					out_channels,
					kernel_size=1,
					stride=stride,
					bias=False,
				),
				nn.BatchNorm2d(out_channels),
			)
		else:
			self.proj = None

	def forward(self, x):
		identity = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
        
        # apply projection to the identity if dimensions don't match
		if self.proj is not None:
			identity = self.proj(identity)

		x = x + identity
		return self.relu(x)


class PocketEncoder(nn.Module):
	def __init__(self, in_channels=1, latent_dim=128):
		super().__init__()
		# resnet-like architecture with 3 residual blocks
		self.stem = nn.Sequential(
			nn.Conv2d(
				in_channels,
				32,
				kernel_size=7,
				stride=2,
				padding=3,
				bias=False,
			),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
		)

		self.layer1 = ResidualBlock(32, 64, stride=2)
		self.layer2 = ResidualBlock(64, 128, stride=2)
		self.layer3 = ResidualBlock(128, 256, stride=2)
        
		self.gap = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(256, latent_dim)

	def forward(self, x):
		x = self.stem(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.gap(x)
		x = torch.flatten(x, 1)
		return self.fc(x)
