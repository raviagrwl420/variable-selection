import itertools

import torch

import torch.nn as nn
import torch.optim as optim

class NNRank(nn.Module):
	"""
	A simple 2 layer neural network for ranking.
	"""

	def __init__(self, n_in, n_hidden, n_out, bias=True):
		super(NNRank, self).__init__()

		self.n_in = n_in
		self.n_hidden = n_hidden

		self.fc1 = nn.Linear(n_in, n_hidden)
		self.dropout = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(n_hidden, n_out)


		self.model = nn.Sequential(self.fc1, nn.ReLU(), self.dropout, self.fc2, nn.Sigmoid())

	def forward(self, inputs):
		return self.model(inputs)

	def train(self, inputs, targets, num_epoch):
		# Initialize criterion as MarginRankingLoss 
		criterion = nn.MarginRankingLoss()

		# Initialize optimizer
		optimizer = optim.SGD(list(self.fc1.parameters()) + list(self.fc2.parameters()),
			lr=0.0001, momentum=0.9, weight_decay=0.01)

		for epoch in range(num_epoch):
			# Zero the parameter gradients
			optimizer.zero_grad()

			# Forward + backward + optimize
			outputs = self.forward(inputs)

			# Select all the items with target class 1 i.e good variables
			outputs1 = torch.index_select(outputs, 0, torch.tensor(targets==1, dtype=torch.long).nonzero().reshape(-1,))
			# Select all the items with target class 0 i.e. bad variables
			outputs0 = torch.index_select(outputs, 0, torch.tensor(targets==0, dtype=torch.long).nonzero().reshape(-1,))
			# Use all combinations of item1 and item2 along with the corresponding targets
			pairs = list(itertools.product(outputs1, outputs0))
			item1 = torch.stack([pair[0] for pair in pairs])
			item2 = torch.stack([pair[1] for pair in pairs])
			modified_targets = torch.ones(len(pairs))

			# item1 is the first item from the pair
			# item2 is the second item from the pair
			# target is either +1 or -1 depending on the ordering of item1 and item2
			loss = criterion(item1, item2, modified_targets)

			loss.backward()
			optimizer.step()

			print('[%d] loss: %.3f' % (epoch + 1, loss.item()))
