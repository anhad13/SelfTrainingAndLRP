import pickle
import sys
sys.path.append(".")
from utils.data_loader import build_tree, get_brackets
import numpy
import torch

def get_stats(outputs):
	numSent = len(outputs[0])
	for x in outputs:
		assert len(x) == numSent
	numReports = len(outputs)
	agree = 0
	fullagree_f1 = []
	av_lenth = {2: [], 3: [], 4: [], 5: []}
	av_f1 = {2: [], 3: [], 4: [], 5: []}
	partial_agree = {2: 0, 3: 0, 4: 0, 5: 0}
	for j in range(numSent):
		brackets = []
		f1s = []
		for i in range(numReports):
			brackets.append(sorted(get_brackets(outputs[i][j]['pred_tree'])[0]))
			f1s.append(outputs[i][j]['f1'])
		for i in range(numReports):
			match_no = 0
			matched_nos =[]
			for k in range(numReports):
				if brackets[k]==brackets[i]:
					match_no += 1
			if match_no < 2 or (match_no in matched_nos):
				continue
			else:
				matched_nos.append(matched_no)
				partial_agree[match_no]+=1
				av_lenth[match_no].append(len(outputs[0][j]['example']))
				av_f1[match_no].append(numpy.mean(f1s))
		if j%1000==0:
			print(str(j)+"-evaluated.")
	for k in av_lenth:
		print("Matching "+str(k)+" reports: "+str(partial_agree[k]))
		print("Average F1 of those sentences: " + str(av_f1[k]))
		print("Av Length of sentences: " + str(av_f1[k]))

# Example call: python scripts
if __name__ == '__main__':
    out_files = sys.argv[1].split(",")
    #loading pickled models.
    outputs = []
    for x in out_files:
    	print("Extracting " + str(x))
    	outputs.append(pickle.load(open(x, "rb")))
    get_stats(outputs)
