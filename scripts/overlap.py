import pickle
import sys
sys.path.append(".")
from utils.data_loader import build_tree, get_brackets
import numpy
import torch
from utils import data_loader

def extract_hm():
	train_data, valid_data, test_data = data_loader_str.main("data/")
	hm = {}
	for i in range(len(train_data[0])):
		hm[" ",join(train_data[4][i])] = train_data[3][i]
	print("Extracted hashmap to goldtrees.")
	return hm

def f1compute(pred_brackets, gold_brackets):
	overlap = pred_brackets.intersection(gold_brackets)    
    prec = float(len(overlap)) / (len(pred_brackets) + 1e-8)
    reca = float(len(overlap)) / (len(gold_brackets) + 1e-8)
    if len(gold_brackets) == 0:
        reca = 1.
        if len(pred_brackets) == 0: 
            prec = 1.
    f1 = 2 * prec * reca / (prec + reca + 1e-8)
    return f1


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
	hm = extract_hm()
	for j in range(numSent):
		brackets = []
		f1s = []
		gold_brackets = hm[" ".join(outputs[0][j]['example'])]
		for i in range(numReports):
			brackets.append(sorted(get_brackets(outputs[i][j]['pred_tree'])[0]))
			pred_brackets = get_brackets(outputs[i][j]['pred_tree'])[0]
			computed_f1 = f1compute(pred_brackets, gold_brackets)
			f1s.append(computed_f1)
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
			not_match_f1s = []
			for el in f1s:
				if el not in best_matchf1s:
					not_match_f1s.append(el)
			av_f1_diff[best_match].append(numpy.mean(best_matchf1s)-numpy.mean(not_match_f1s))
	for k in av_lenth:
		print("Matching "+str(k)+" reports: "+str(partial_agree[k]))
		print("Average F1 of those sentences: " + str(numpy.mean(av_f1[k])))
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
