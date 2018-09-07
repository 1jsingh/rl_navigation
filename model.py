import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
	"""Actor (Policy) Model."""
	def __init__(self, state_size, action_size, seed):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
		"""
		super(QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		
		# fc layers
		self.fc1 = nn.Linear(state_size,32)
		self.fc2 = nn.Linear(32,64)
		self.fc3 = nn.Linear(64,128)
		self.out = nn.Linear(128,action_size)
		
	def forward(self, state):
		"""Build a network that maps state -> action values."""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.out(x)
		return x