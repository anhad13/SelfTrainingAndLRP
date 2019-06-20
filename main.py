import sys
from random import shuffle
import numpy
import torch
import torch.optim as optim

from utils import data_loader
from model.parser import Parser
from utils.data_loader import build_tree, get_brackets


def make_tuples(dataset):
    new_data = []
    for x, y in zip(dataset[0], dataset[1]):
        new_data.append((x, y))
    return(new_data)


def ranking_loss(pred, gold):
    loss = 0.
    for i in range(0, pred.shape[1]):
        loss = loss + ((gold[:, i] - pred[:, i]) ** 2).sum()
#        for j in range(i+1, pred.shape[1]):  # assuming target_dist has same length at pred_dist
#            t_dist = gold[:,i] - gold[:,j]
#            p_dist = pred[:,i] - pred[:,j]
#            possible_loss = (t_dist - p_dist) ** 2
##            signed = torch.sign(t_dist) * p_dist
##            possible_loss = 1.0 * (0. - signed)
##            possible_loss = torch.clamp(possible_loss, 0.0, 1000.0)
#            loss += torch.mean(possible_loss)
    return loss


def eval_fct(model, dataset):
    model.eval()
    prec_list = []
    reca_list = []
    f1_list = []
    #for i in range(len(dataset[0])):
    for i in range(1):
        x = dataset[0][i]
        y = dataset[1][i]
        gold_brackets = dataset[3][i]
        sent = dataset[4][i]

        preds = model(x)
        print(y)
        print(preds)
        print()
        pred_tree = build_tree([-1] + list(preds.data[0]), sent)
        pred_brackets = get_brackets(pred_tree)[0]

        overlap = pred_brackets.intersection(gold_brackets)
        
        prec = float(len(overlap)) / (len(pred_brackets) + 1e-8)
        reca = float(len(overlap)) / (len(gold_brackets) + 1e-8)
        if len(gold_brackets) == 0:
            reca = 1.
            if len(pred_brackets) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1)
    
    # TODO: This is very weird for F1, reconsider.
    return numpy.mean(f1_list)


def train_fct(train_data, valid_data, vocab, nemb=100, nhid=300, epochs=300):
    model = Parser(nemb, nhid, len(vocab))
    optimizer = optim.Adam(model.parameters())
    train = make_tuples(train_data)
    print(len(train))
    
    for epoch in range(epochs):
        model.train()
        av_loss = 0.
        shuffle(train)
        for (x, y) in train:
            optimizer.zero_grad()
            preds = model(x)
            loss = ranking_loss(preds, y.unsqueeze(0))
            av_loss += loss
            loss.backward()
            optimizer.step()
        av_loss /= len(train)
        
        f1 = eval_fct(model, train_data)
        
        print('Epoch: ' + str(epoch))
        print('Loss: ' + str(av_loss.data))
        print('F1: ' + str(f1))
    return None


if __name__ == '__main__':
    '''
       For now, the first argument is the path to the data.
       # TODO: use a reasonable argument parser!
    '''
    train_data, valid_data, test_data = data_loader.main(sys.argv[1])
    train_fct(train_data, valid_data, valid_data[-1])
