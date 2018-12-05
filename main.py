# Main file

import torch
import itertools

import numpy as np

from src.lop import lop
from src.nn_rank import NNRank

# lop('RandB/N-p40-01').solve_instance()

model = NNRank(100, 5, 1)

inputs = torch.randn(10000).reshape(100, 100)
targets = torch.cat((torch.zeros(50), torch.ones(50)), dim=0)

model.train(inputs, targets, 100)
