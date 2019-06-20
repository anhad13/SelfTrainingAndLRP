import sys
import torch
import torch.optim as optim

from utils import data_loader
from model.parser import Parser


def make_tuples(dataset):
    new_data = []
    for x, y in zip(dataset[0], dataset[1]):
        new_data.append((x, y))
    return(new_data)


def ranking_loss(pred, gold):
    loss = 0.
    for i in range(0, pred.shape[1]):
        for j in range(i+1, pred.shape[1]):  # assuming target_dist has same length at pred_dist
            t_dist = gold[:,i] - gold[:,j]
            p_dist = pred[:,i] - pred[:,j]
            signed = torch.sign(t_dist) * p_dist
            possible_loss = 1.0 *(1. - signed)
            possible_loss = torch.clamp(possible_loss, 0.0, 1000.0)
            loss += torch.mean(possible_loss)
    return loss


def train_fct(train, valid, vocab, nemb=11, nhid=13, epochs=100):
    model = Parser(nemb, nhid, len(vocab))
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        av_loss = 0.
        for (x, y) in train:
            optimizer.zero_grad()
            preds = model(x).squeeze(2)
            loss = ranking_loss(preds, y.unsqueeze(0))
            av_loss += loss
            loss.backward()
            optimizer.step()
        av_loss /= len(train)
        print('Epoch :' + str(epoch))
        print('Loss: ' + str(av_loss))
    return None


if __name__ == '__main__':
    '''
       For now, the first argument is the path to the data.
       # TODO: use a reasonable argument parser!
    '''
    train_data, valid_data, test_data = data_loader.main(sys.argv[1])
    train = make_tuples(train_data)
    valid = make_tuples(valid_data)
    train_fct(train, valid, valid_data[-1])
