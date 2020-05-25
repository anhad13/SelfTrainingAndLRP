import pickle
import sys
sys.path.append(".")
from utils.data_loader import build_tree, get_brackets
import numpy
import torch
from utils import data_loader_legacy as data_loader
from utils.tree_to_gate import tree_to_gates
def extract_hm():
	train_data, valid_data, test_data = data_loader.main("data/")
	hm = {}
	for i in range(len(train_data[0])):
		hm[" ".join(train_data[4][i])] = [train_data[0][i], train_data[1][i], train_data[2][i], train_data[3][i], train_data[4][i], train_data[5][i]]
	print("Extracted hashmap to goldtrees.")
	return hm, train_data[-1]

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

def list2distance(tree_ar):
    if type(tree_ar) != list:
        return [], 0
    ld, lh = list2distance(tree_ar[0])
    rd, rh = list2distance(tree_ar[1])
    hh = max([lh, rh])+1
    return ld + [hh] + rd, hh

def depth(arr):
	if type(arr)!=list:
		return 0
	return max(depth(arr[0]),depth(arr[1]))+1

def get_stats(outputs):
	numSent = len(outputs[0])
	for x in outputs:
		assert len(x) == numSent
	numReports = len(outputs)
	agree = 0
	fullagree_f1 = []
	av_lenth = {1: [], 2: [], 3: [], 4: [], 5: [],6: [],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[],20:[]}
	av_depth = {1: [], 2: [], 3: [], 4: [], 5: [],6: [],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[],20:[]}	
	av_f1 = {1: [], 2: [], 3: [], 4: [], 5: [],6: [],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[],20:[]}
	av_f1_diff = {1:[],2: [], 3: [], 4: [], 5: [],6: [],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[],20:[]}
	partial_agree = {1: 0,2: 0, 3: 0, 4: 0, 5: 0,6: 0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0}
	av_d = []
	todump = [[],[],[],[],[],[],[]]
	hm, vocab = extract_hm()
	for j in range(numSent):
		brackets = []
		f1s = []
		fulldata =  hm[" ".join(outputs[0][j]['example'])]
		gold_brackets = fulldata[3]
		for i in range(numReports):
			brackets.append(sorted(get_brackets(outputs[i][j]['pred_tree'])[0]))
			pred_brackets = get_brackets(outputs[i][j]['pred_tree'])[0]
			computed_f1 = f1compute(pred_brackets, gold_brackets)
			f1s.append(computed_f1)
		best_match = 1
		best_matchf1s=None
		besttree = None	
		for i in range(numReports):
			match_no = 0
			matchf1s = []
			predtree = None
			for k in range(numReports):
				if brackets[k]==brackets[i]:
					match_no += 1
					matchf1s.append(f1s[k])
					predtree = outputs[i][j]['pred_tree']
			if best_match < match_no:
				best_match = match_no 
				best_matchf1s = matchf1s
				besttree = predtree
		if best_match > 1:
			partial_agree[best_match]+=1
			av_lenth[best_match].append(len(outputs[0][j]['example']))
			av_f1[best_match].append(numpy.mean(best_matchf1s))
			av_depth[best_match].append(depth(predtree))
			not_match_f1s = []
			for el in f1s: 
				if el not in best_matchf1s:
					not_match_f1s.append(el)
			av_f1_diff[best_match].append(numpy.mean(best_matchf1s)-numpy.mean(not_match_f1s))
			if best_match >= 3:
				for i in [4]:
					todump[i].append(fulldata[i])
				todump[0].append(torch.LongTensor(fulldata[0]))         
				todump[1].append(torch.FloatTensor(list2distance(besttree)[0]))
				todump[2].append(besttree)
				todump[3].append(get_brackets(besttree)[0])
				todump[5].append(torch.FloatTensor(tree_to_gates(besttree)))
		else:
			av_lenth[best_match].append(len(outputs[0][j]['example']))
			av_f1[best_match].append(numpy.mean(f1s))
			av_depth[best_match].append(depth(predtree))
			partial_agree[1]+=1
	todump[6] = vocab
	pickle.dump(todump, open("10_5_3.data", "wb"))

	for k in av_lenth:
		print("Matching "+str(k)+" reports: "+str(partial_agree[k]))
		print("Average F1 of those sentences: " + str(numpy.mean(av_f1[k])))
		print("Av Length of sentences: " + str(numpy.mean(av_lenth[k])))
		print("Min Length: "+str(numpy.min(av_lenth[k])))
		print("Max Length: "+str(numpy.max(av_lenth[k])))
		print("Av Depth of sentences: " + str(numpy.mean(av_depth[k])))

# Example call: python scripts
if __name__ == '__main__':
    out_files = sys.argv[1].split(",")
    #loading pickled models.
    outputs = []
    for x in out_files:
    	print("Extracting " + str(x))
    	outputs.append(pickle.load(open(x, "rb")))
    get_stats(outputs)
