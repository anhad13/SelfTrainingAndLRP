import sys, os
import re

import numpy
import torch
import nltk
from nltk.corpus import ptb
from nltk.corpus import BracketParseCorpusReader


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
        if tree.label() in word_tags:
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
        if tag in word_tags:
            w = w.lower()
            w = re.sub('[0-9]+', 'N', w)
            # if tag == 'CD':
            #     w = 'N'
            words.append(w)
    return words


def load_trees(ids, vocab=None, grow_vocab=True):
    '''
       This returns
       1) a list of torch.LongTensors containing the indices of all not filtered words of each sentence
       2) a torch.FloatTensor containing the corresponding distances between words
       3) the original sentence with in bracket format
       4) the brackets as tuples
    '''
    if not vocab:
        vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    all_sents, all_trees, all_dists, all_brackets, all_words = [], [], [], [], []
    for id in ids:
        #sentences = ptb.parsed_sents(id)
        ptb = BracketParseCorpusReader('', id)
        for sent in ptb.parsed_sents():
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
            nltk.treetransforms.chomsky_normal_form(sent)
            treelist = tree2list(sent)
            brackets = get_brackets(treelist)[0]

            all_sents.append(torch.LongTensor(idx))
            all_trees.append(treelist)
            all_dists.append(torch.FloatTensor(list2distance(treelist)[0]))
            all_brackets.append(brackets)
            all_words.append(words)

    return all_sents, all_dists, all_trees, all_brackets, all_words, vocab


def main(path):
    train_file_ids = []
    valid_file_ids = []
    test_file_ids = []
    rest_file_ids = []
    for (path, _, fnames) in os.walk(path):
        for fname in fnames:
            id = path + '/' + fname
            if 'data/wsj/00/wsj_0000.mrg' <= id <= 'data/wsj/21/wsj_2199.mrg':
                train_file_ids.append(id)
            elif 'data/wsj/22/wsj_2200.mrg' <= id <= 'data/wsj/22/wsj_2299.mrg':
                valid_file_ids.append(id)
            elif 'data/wsj/23/wsj_2300.mrg' <= id <= 'data/wsj/23/wsj_2399.mrg':
                test_file_ids.append(id)
            elif 'data/wsj/00/wsj_0000.mrg' <= id <= 'data/wsj/01/wsj_0199.mrg' or 'data/wsj/24/wsj_2400.mrg' <= id <= 'data/wsj/24/wsj_2499.mrg':
                rest_file_ids.append(id)

    train_data = load_trees(train_file_ids[:1])
    valid_data = load_trees(valid_file_ids[:1], vocab=train_data[-1], grow_vocab=True)
    test_data = load_trees(test_file_ids[:1], vocab=train_data[-1], grow_vocab=False)
    rest_data = load_trees(rest_file_ids[:1], vocab=train_data[-1], grow_vocab=False)
    number_sentences = len(train_data[0]) + len(valid_data[0]) + len(test_data[0]) + len(rest_data[0])
    print('Number of sentences loaded: ' + str(number_sentences))
    
    return train_data, valid_data, test_data


if __name__ == '__main__':
    train_data, valid_data, test_data = main(sys.argv[1])

