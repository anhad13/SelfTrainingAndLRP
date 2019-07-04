import pickle
import sys
sys.path.append(".")
from utils.data_loader import build_tree, get_brackets
import numpy
import torch
# p1 = pickle.load(open("UNSUP_0000", "rb"))
# p2 = pickle.load(open("UNSUP_1111", "rb"))
# p3 = pickle.load(open("UNSUP_2222", "rb"))
# total = 0
# agree = 0
# buckets = {0: 0, 0.25: 0, 0.5:0, 0.75:0}
# for x in zip(p1, p2, p3):
# 	overlap = x[0].intersection(x[1])
# 	overlap = x[2].intersection(overlap)
# 	max_len = max([len(y) for y in x])
# 	if max_len == 0:
# 		continue
# 	total += 1
# 	overlap_p = float(len(overlap))/float(max_len)
# 	if overlap_p < 0.25:
# 		buckets[0] += 1
# 	elif overlap_p < 0.5:
# 		buckets[0.25] += 1
# 	elif overlap_p < 0.75:
# 		buckets[0.5] += 1
# 	else:
# 		buckets[0.75] += 1
# 	if sorted(x[0]) == sorted(x[1]) == sorted(x[1]):
# 		agree += 1
# print(agree)
# print(total)
# print(buckets)

def get_stats(outputs):
	numSent = len(outputs[0])
	for x in outputs:
		assert len(x) == numSent
	numReports = len(outputs)
	agree = 0
	fullagree_f1 = [];av_lenth=[]
	for j in range(numSent):
		overlap = None
		avf1 = []
		for i in range(numReports):
			avf1.append(outputs[i][j]['f1'])
			if not overlap:
				overlap = get_brackets(outputs[i][j]['pred_tree'])[0]
			else:
				overlap = overlap.intersection(get_brackets(outputs[i][j]['pred_tree'])[0])
		if sorted(overlap) == sorted(get_brackets(outputs[0][j]['pred_tree'])[0]):
			agree += 1
			fullagree_f1.append(numpy.mean(avf1));av_lenth.append(len(outputs[0][j]['example']))
	print("Number of full agreements: " + str(agree) + " / "+ str(numSent) +" with av length: "+str(numpy.average(av_lenth))+" with av F1: " + str(numpy.average(fullagree_f1)))




# Example call: python scripts
if __name__ == '__main__':
    out_files = sys.argv[1].split(",")
    #loading pickled models.
    outputs = []
    for x in out_files:
    	print("Extracting " + str(x))
    	outputs.append(pickle.load(open(x, "rb")))
    get_stats(outputs)
