from PYEVALB import scorer
from PYEVALB import parser
import numpy
import nltk
import re

def compute_f1(tokens, gold_nltk_tree, leaf, nonleaf, pred_tree, pos_list, gold_leafs, gold_nonleafs, gold_dists,  hmap):
	"""

	"""
	sent = tokens[1:-1]
	gold = " ".join(str(gold_nltk_tree).split())
	leafa = []
	for x in leaf[1:-1]:
		leafa.append(hmap[int(x)])
	nonleafa = []
	for x in nonleaf:
		nonleafa.append(hmap[int(x)])
	gleafa = []
	for x in gold_leafs:
		gleafa.append(hmap[int(x)])
	gnonleafa = []
	for x in gold_nonleafs:
		gnonleafa.append(hmap[int(x)])
	gold = build_tree_labelled(gold_dists, sent, gnonleafa, gleafa, pos_list)
	test = build_tree_labelled(pred_tree, sent, nonleafa, gleafa, pos_list)
	gold_tree = parser.create_from_bracket_string(gold)
	test_tree = parser.create_from_bracket_string(test)#;print(gold_tree);print("\n");print(test_tree)
	res = scorer.Scorer().score_trees(gold_tree, test_tree)
	f1 = 2 * res.prec * res.recall / (res.prec + res.recall + 1e-8)
	return f1

def tree2list(tree, parent_arc=[]):
    if isinstance(tree, nltk.Tree):
        label = tree.label()
        if isinstance(tree[0], nltk.Tree):
            label = re.split('-|=', tree.label())[0]
        root_arc_list = parent_arc + [label]
        root_arc = '+'.join(root_arc_list)
        if len(tree) == 1:
            root, arc, tag = tree2list(tree[0], parent_arc=root_arc_list)
        elif len(tree) == 2:
            c0, arc0, tag0 = tree2list(tree[0])
            c1, arc1, tag1 = tree2list(tree[1])
            root = [c0, c1]
            arc = arc0 + [root_arc] + arc1
            tag = tag0 + tag1
        else:
            c0, arc0, tag0 = tree2list(tree[0])
            c1, arc1, tag1 = tree2list(nltk.Tree('<empty>', tree[1:]))
            if bin == 0:
                root = [c0] + c1
            else:
                root = [c0, c1]
            arc = arc0 + [root_arc] + arc1
            tag = tag0 + tag1
        return root, arc, tag
    else:
        if len(parent_arc) == 1:
            parent_arc.insert(0, '<empty>')
        # parent_arc[-1] = '<POS>'
        del parent_arc[-1]
        return str(tree), [], ['+'.join(parent_arc)]


def build_tree_labelled(depth, sen, label_nonleaf, label_leaf, pos_list):
    assert len(depth) == len(sen) - 1
    if len(sen)==1:
    	assert len(label_leaf) == 1
    	if label_leaf[0] == "phi":#phi allert
    		return "("+pos_list[0]+" "+sen[0]+")" 
    	else:
    		return "("+label_leaf[0] + "("+pos_list[0]+" "+sen[0]+"))"
    idx_max = numpy.argmax(depth)
    curr_label = label_nonleaf[idx_max]
    lt = build_tree_labelled(depth[:idx_max],sen[:idx_max+1],label_nonleaf[:idx_max], label_leaf[:idx_max+1], pos_list[:idx_max+1])
    rt = build_tree_labelled(depth[idx_max+1:],sen[idx_max+1:],label_nonleaf[idx_max+1:], label_leaf[idx_max+1:], pos_list[idx_max+1:])
    return "("+curr_label+" "+lt+" "+rt+")"
