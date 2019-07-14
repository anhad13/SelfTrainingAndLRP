import pickle
import sys
sys.path.append(".")
from utils import data_loader
import numpy as np
from random import shuffle
import torch

def extract(arr, index_list):
	final_arr = []
	for i in index_list:
		final_arr.append(arr[i])
	return final_arr

# Example call: python scripts
if __name__ == '__main__': #each will have 60% of the total data.
	no_models = int(sys.argv[1])
	percentage_take = 0.7
	train_data , _ , _ = data_loader.main("data/")
	train_data = np.array(train_data)
	lentrain = len(train_data[0])
	eachm = int(lentrain * percentage_take)
	exhaustive_portion = int(len(train_data[0])/no_models)
	# common portion is comming for all data points.
	indices = np.arange(lentrain)
	# extract portions for making it exhaustive
	models = []
	for i in range(no_models):
		tmp = []
		for j in range(len(train_data)-1):
			tmp.append(extract(train_data[j],indices[i*exhaustive_portion:(i+1)*exhaustive_portion]))
		models.append(tmp)
	for i in range(no_models):
		shuffle(indices)
		for j in range(len(train_data)-1):
			models[i][j]+=extract(train_data[j], indices[:eachm])
		models[i].append(train_data[-1])
	# now dump everything.
	print("Dumping....")
	for i in range(no_models):
		pickle.dump(models[i], open("scripts/0.7_model_"+str(i))




