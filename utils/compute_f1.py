from PYEVALB import scorer
from PYEVALB import parser
import numpy
import nltk

def compute_f1(tokens, gold_nltk_tree, leaf, nonleaf, pred_tree, hmap):
	"""

	"""
	sent = tokens[1:-1]
	gold = tree2labellist(gold_nltk_tree)
	leafa = []
	for x in leaf[1:-1]:
		leafa.append(hmap[int(x)])
	nonleafa = []
	for x in nonleaf:
		nonleafa.append(hmap[int(x)])
	test = build_tree_labelled(pred_tree, sent, nonleafa, leafa)
	gold_tree = parser.create_from_bracket_string(gold)
	test_tree = parser.create_from_bracket_string(test)
	res = scorer.Scorer().score_trees(gold_tree, test_tree)
	f1 = 2 * res.prec * res.recall / (res.prec + res.recall + 1e-8)
	return f1


def build_tree_labelled(depth, sen, label_nonleaf, label_leaf):
    assert len(depth) == len(sen) - 1
    if len(sen)==1:
    	assert len(label_leaf) == 1
    	return "("+label_leaf[0]+" "+sen[0]+")"
    idx_max = numpy.argmax(depth)
    curr_label = label_nonleaf[idx_max]
    lt = build_tree_labelled(depth[:idx_max],sen[:idx_max+1],label_nonleaf[:idx_max], label_leaf[:idx_max+1])
    rt = build_tree_labelled(depth[idx_max+1:],sen[idx_max+1:],label_nonleaf[idx_max+1:], label_leaf[idx_max+1:])
    return "("+curr_label+" "+lt+" "+rt+")"

def tree2labellist(tree):
    if isinstance(tree, nltk.Tree):
        if len(tree.leaves())<=1:
        	return "("+tree.label()+" "+tree.leaves()[0]+")"
        elif len(tree)==1:
            return tree2labellist(tree[0])
        current = tree.label().split("|")[-1]
        c1 = tree2labellist(tree[0])
        c2 = tree2labellist(tree[1])
        return "("+current+" "+c1+" "+c2+")"
    return ""