import sys, os
import re

import numpy
import torch
import nltk
from nltk.corpus import ptb
from nltk.corpus import BracketParseCorpusReader
from utils.tree_to_gate import tree_to_gates
import pickle
from nltk import Tree

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']


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
        w = w.lower()
        w = re.sub('[0-9]+', 'N', w)
        # if tag == 'CD':
        #     w = 'N'
        words.append(w)
    return words

def parse_arabic_sents(fname):
    trees = []
    current_sent = ''
    brackets = 0
    for line in open(fname):
        line = line.strip()
        if line == '' or line[0] == '%':
            continue
        for word in line.split(' '):
            for char in word:
                if char == '(':
                    brackets += 1
                elif char == ')':
                    brackets -= 1
            current_sent += ' ' + word
            if brackets == 0:
                if False:
                    current_sent = current_sent[1:]
                    new_sent = []
                    for p in current_sent.split(' '):
                        if p[-1] == ')' and not '(' in p:
                            new_sent.append('arabic_' + p)
                        else:
                            new_sent.append(p)
                    current_sent = ' '.join(new_sent)
                trees.append(Tree.fromstring(current_sent))
                current_sent= ""
    return trees

def load_trees(path, vocab=None, grow_vocab=True, supervision_limit=-1, supervised_model=False, semisupervised=False, binarize=False):
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
    trees = []
    for directory, dirnames, filenames in os.walk(path):
        for file in filenames:
            trees += parse_arabic_sents(path+'/'+file)
    for id in [1]:
        for sent in trees:
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
                nltk.treetransforms.chomsky_normal_form(sent)
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


def main(path, supervision_limit=-1, supervised_model=False, vocabulary=None, pickled_file_path=None, bagging=False, semisupervised=False, force_binarize=False):
    path = "data/arabic/"
    if pickled_file_path == None:
        train_data = load_trees(path + '26', vocab=vocabulary, grow_vocab= (vocabulary==None), supervision_limit=supervision_limit, supervised_model=supervised_model, semisupervised=semisupervised, binarize=True)
    else: # assumption: supervised load from pickle and all data is UNSUP
        pickled_training_data = pickle.load(open(pickled_file_path, "rb"))
        if bagging:
            print("INIT TO ZERO...")
            train_data = [[],[],[],[],[],[],[], pickled_training_data[-1]]
        else:
            train_data = load_trees(path + '26', vocab=pickled_training_data[-1], grow_vocab= False, binarize=True)
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
    valid_data = load_trees(path + '27', vocab=train_data[-1], grow_vocab= (vocabulary==None), binarize= force_binarize)
    test_data = load_trees(path + '28', vocab=train_data[-1], grow_vocab=False, binarize= force_binarize)
    #rest_data = load_trees(rest_file_ids[:1], vocab=train_data[-1], grow_vocab=False)
    rest_data = []
    number_sentences = len(train_data[0]) + len(valid_data[0]) + len(test_data[0])
    print('Number of sentences loaded: ' + str(number_sentences))
    
    return train_data, valid_data, test_data


if __name__ == '__main__':
    train_data, valid_data, test_data = main(sys.argv[1])
