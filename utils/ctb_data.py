import sys, os
import re

import numpy
import torch
import nltk
from nltk.corpus import ptb
from nltk.corpus import BracketParseCorpusReader
from utils.tree_to_gate import tree_to_gates
import pickle
from os.path import isfile, join, isdir
from os import listdir

def list2distance(tree_ar):
    if type(tree_ar) != list:
        return [], 0
    ld, lh = list2distance(tree_ar[0])
    rd, rh = list2distance(tree_ar[1])
    hh = max([lh, rh])+1
    return ld + [hh] + rd, hh


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def tree2list(tree):
    if isinstance(tree, nltk.Tree):
        if len(tree.leaves()) == 1:
            return tree.leaves()[0]
        else:
            root = []
            for child in tree:
                c = tree2list(child)
                if c != []:
                    root.append(c)
            if len(root) > 1:
                return root
            elif len(root) == 1:
                return root[0]
    return []


def build_tree(depth, sen):
    assert len(depth) == len(sen) - 1
    if len(sen)==1 or len(sen) == 2:
        return sen
    parse_tree=[]
    idx_max = numpy.argmax(depth)
    if len(sen[:idx_max+1]) >= 1:
        parse_tree.append(build_tree(depth[:idx_max],sen[:idx_max+1]))
    if len(sen[idx_max+1:]) >= 1:
        parse_tree.append(build_tree(depth[idx_max+1:],sen[idx_max+1:]))
    return parse_tree


def filter_words(tree):
    words = []
    for w, tag in tree.pos():
        words.append(w)
    return words


def checkoserrror(path):
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def load_trees(path, ids, vocab=None, semisupervised=False, grow_vocab=True, supervision_limit=-1, supervised_model=False, binarize = False):
    '''
       This returns
       1) a list of torch.LongTensors containing the indices of all not filtered words of each sentence
       2) a torch.FloatTensor containing the corresponding distances between words
       3) the original sentence with in bracket format
       4) the brackets as tuples
       5) the gate values for training PRPN in a supervised way
    '''
    if not vocab:
        vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    all_sents, all_trees, all_dists, all_brackets, all_words, all_gates, skip_sup = [], [], [], [], [], [], []
    counter = 0
    files = []
    dontexist = 0
    ctb = convert_ctb5_to_backeted(path, ids)
    # for fid in ids:
    #     # f = 'chtb_%03d.fid' % fid
    #     # if fid > 1000:
    #     #     f = 'chtb_%04d.fid' % fid
    #     f = fid
    #     try:
    #         ctb.parsed_sents()[f]
    #         files.append(f)
    #     except OSError as exception:
    #         dontexist +=1
    for id in [1]:
        for sent in ctb.parsed_sents():
            words = ['<bos>'] + filter_words(sent) + ['<eos>']
            idx = []
            for word in words:
                if word not in vocab:
                    if grow_vocab:
                        vocab[word] = len(vocab)
                    else:
                        word = '<unk>'
                idx.append(vocab[word])
            if len(words)<=3:
                continue
                print("skipping")
            # Binarize tree.
            if binarize:
                try:
                    nltk.treetransforms.chomsky_normal_form(sent)
                except:
                    print(sent)
                    continue
            treelist = tree2list(sent)
            gate_values = tree_to_gates(treelist)
            brackets = get_brackets(treelist)[0]
            if supervision_limit > -1 and counter >= supervision_limit:
                if (not semisupervised) and supervised_model:
                    break
                all_dists.append(torch.zeros_like(torch.FloatTensor(list2distance(treelist)[0])))
                all_gates.append(torch.zeros_like(torch.FloatTensor(gate_values)))
                skip_sup.append(False)
            else:
                all_dists.append(torch.FloatTensor(list2distance(treelist)[0]))
                all_gates.append(torch.FloatTensor(gate_values))
                if semisupervised==False and supervised_model==False:
                   skip_sup.append(False)
                else:
                   skip_sup.append(True)
            all_sents.append(torch.LongTensor(idx))
            all_trees.append(treelist)
            all_brackets.append(brackets)
            all_words.append(words)
            counter += 1
        if supervision_limit > -1 and counter >= supervision_limit and supervised_model:
            break
    return all_sents, all_dists, all_trees, all_brackets, all_words, all_gates, skip_sup, vocab


def convert_ctb5_to_backeted(ctb_root, ids):
    ctb_root = join(ctb_root, 'bracketed')
    fids = [f for f in listdir(ctb_root) if isfile(join(ctb_root, f)) and f.endswith('.fid')]
    lines = []
    files = []
    out = open(ctb_root+"files.ctb", "w")
    for fid in ids:
        f = 'chtb_%03d.fid' % fid
        if fid > 1000:
            f = 'chtb_%04d.fid' % fid
        files.append(f)
    doesnt_exist = 0
    for f in files:
        if not os.path.exists(join(ctb_root, f)):
            doesnt_exist+=1
            continue
        with open(join(ctb_root, f), encoding='GB2312') as src:
            in_s_tag = False
            try:
                for line in src:
                    if line.startswith('<S ID='):
                        in_s_tag = True
                    elif line.startswith('</S>'):
                        in_s_tag = False
                    elif in_s_tag:
                        out.write(line)
            except:
                # The last file throws encoding error at the very end, doesn't affect sentences.
                pass
    print(doesnt_exist)
    ctb = BracketParseCorpusReader('' , ctb_root+'files.ctb')
    return ctb



def main(data = 'data/ctb/',supervision_limit=-1, supervised_model=False, vocabulary=None, pickled_file_path=None, bagging=False, semisupervised=False, force_binarize=False):
    path = "data/ctb/"
    #ctb = convert_ctb5_to_backeted(path)
    training = list(range(1, 815 + 1)) + list(range(1001, 1136 + 1))
    development = list(range(886, 931 + 1)) + list(range(1148, 1151 + 1))
    test = list(range(816, 885 + 1)) + list(range(1137, 1147 + 1))
    if pickled_file_path == None:
        train_data = load_trees(path, training, vocab=vocabulary, grow_vocab=(vocabulary==None), supervision_limit=supervision_limit, supervised_model=supervised_model, semisupervised=semisupervised, binarize = True)
    else: # assumption: supervised load from pickle and all data is UNSUP
        pickled_training_data = pickle.load(open(pickled_file_path, "rb"))
        train_data = load_trees(train_file_ids, vocab=pickled_training_data[-1], grow_vocab= False, binarize = True)
        print(str(len(pickled_training_data[0]))+"_______LEN")
        for i in range(len(pickled_training_data[0])):
            train_data[0].append(pickled_training_data[0][i])
            train_data[1].append(pickled_training_data[1][i])
            train_data[2].append(pickled_training_data[2][i])
            train_data[3].append(pickled_training_data[3][i])
            train_data[4].append(pickled_training_data[4][i])
            train_data[5].append(pickled_training_data[5][i])
            if not bagging:
                train_data[6].append(True)
            else:
                train_data[6].append(False)
        vocabulary = pickled_training_data[-1]
    valid_data = load_trees(path, development, vocab=train_data[-1], grow_vocab= (vocabulary==None), binarize= force_binarize)
    test_data = load_trees(path, test, vocab=train_data[-1], grow_vocab=False, binarize= force_binarize)
    number_sentences = len(train_data[0]) + len(valid_data[0]) + len(test_data[0]) 
    print('Number of sentences loaded: ' + str(number_sentences))
    
    return train_data, valid_data, test_data


if __name__ == '__main__':
    train_data, valid_data, test_data = main(sys.argv[1])
