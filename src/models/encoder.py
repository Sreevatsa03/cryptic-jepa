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

		conv1 = self.conv1(x)
		bn1 = self.bn1(conv1)
		relu = self.relu(bn1)

		conv2 = self.conv2(relu)
		bn2 = self.bn2(conv2)
        
        # apply projection to the identity if dimensions don't match
		if self.proj is not None:
			identity = self.proj(identity)

		out = bn2 + identity
		out = self.relu(out)
		return out


class ContextEncoder(nn.Module):
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
		conv = self.stem(x)
		rl1 = self.layer1(conv)
		rl2 = self.layer2(rl1)
		rl3 = self.layer3(rl2)
		gap = self.gap(rl3)
		flat = torch.flatten(gap, 1)
		out = self.fc(flat)
		return out
