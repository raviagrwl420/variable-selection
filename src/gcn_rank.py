import torch.nn as nn
import torch.nn.functional as F

from gcn_layer import GCNLayer

class GCNRank(nn.Module):
	def __init__(self, n_in, n_hidden, n_out):
		super(GCNRank, self).__init__()

		self.gc1 = GCNLayer(n_in, n_hidden)
		self.dropout = nn.Dropout(p=0.5)
		self.gc2 = GCNLayer(n_hidden, n_out)

		# self.model = nn.Sequential(self.gc1, nn.ReLU(), self.dropout, self.gc2, self.Sigmoid())

	def forward(self, inputs, adj)
		inputs = F.relu(self.gc1(inputs, adj))
		inputs = F.dropout(inputs, self.dropout, training=self.training)
		inputs = self.gc2(inputs, adj)
		return F.sigmoid(inputs)

	def train(self, inputs, adj)
		# TODO: Add the `train` method from NNRank 
		print('Does nothing!') 