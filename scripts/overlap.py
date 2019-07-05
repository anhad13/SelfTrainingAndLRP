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
	av_f1_diff = {2: [], 3: [], 4: [], 5: []}
	partial_agree = {2: 0, 3: 0, 4: 0, 5: 0}
	for j in range(numSent):
		brackets = []
		f1s = []
		for i in range(numReports):
			brackets.append(sorted(get_brackets(outputs[i][j]['pred_tree'])[0]))
			f1s.append(outputs[i][j]['f1'])
		best_match = 1
		best_matchf1s=None
		for i in range(numReports):
			match_no = 0
			matchf1s=[]			
			for k in range(numReports):
				if brackets[k]==brackets[i]:
					match_no += 1
					matchf1s.append(f1s[k])
			if best_match < match_no:
				best_match = match_no
				best_matchf1s = matchf1s
		if best_match > 1:
			partial_agree[best_match]+=1
			av_lenth[best_match].append(len(outputs[0][j]['example']))
			av_f1[best_match].append(numpy.mean(best_matchf1s))
			av_f1_diff[best_match].append(numpy.mean(best_matchf1s)-numpy.mean(f1s))
	for k in av_lenth:
		print("Matching "+str(k)+" reports: "+str(partial_agree[k]))
		print("Average F1 of those sentences: " + str(numpy.mean(av_f1[k]))+" .. av diff: "+str(numpy.mean(av_f1_diff[k])))
		print("Av Length of sentences: " + str(numpy.mean(av_lenth[k])))
		print("Min Length: "+str(numpy.min(av_lenth[k])))
		print("Max Length: "+str(numpy.max(av_lenth[k])))

# Example call: python scripts
if __name__ == '__main__':
    out_files = sys.argv[1].split(",")
    #loading pickled models.
    outputs = []
    for x in out_files:
    	print("Extracting " + str(x))
    	outputs.append(pickle.load(open(x, "rb")))
    get_stats(outputs)
