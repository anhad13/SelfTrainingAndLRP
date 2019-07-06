import pickle
import sys
sys.path.append(".")
from utils import data_loader
import numpy
import torch

train_data, valid_data, test_data = data_loader_str.main("data/")
hm = {}
for i in range(len(train_data[0])):
	 hm[" ",join(train_data[4][i])] = train_data[3][i]
