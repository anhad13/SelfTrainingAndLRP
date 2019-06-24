import sys
from random import shuffle
import numpy
import torch
import torch.optim as optim

from utils import data_loader
from model.parser import Parser
from utils.data_loader import build_tree, get_brackets


def ranking_loss(pred, gold, mask):
    loss = 0.
    for i in range(0, pred.shape[1]):
    #masked = ((gold[:, i] - pred[:, i]) ** 2) * mask[:, i].float()
    #loss = loss + masked.sum()
        for j in range(i+1, pred.shape[1]):  # assuming target_dist has same length at pred_dist
            t_dist = gold[:,i] - gold[:,j]
            p_dist = pred[:,i] - pred[:,j]
            possible_loss = (t_dist - p_dist) ** 2
            signed = torch.sign(t_dist) * p_dist
            possible_loss = 1.0 * (1. - signed)
            possible_loss = torch.clamp(possible_loss, 0.0, 1000.0)
            possible_loss = possible_loss * mask[:,j].float()
            loss += torch.mean(possible_loss)
    return loss


def eval_fct(model, dataset):
    model.eval()
    prec_list = []
    reca_list = []
    f1_list = []
    for i in range(len(dataset[0])):
        #for i in range(1):
        x = dataset[0][i]
        y = dataset[1][i]
        gold_brackets = dataset[3][i]
        sent = dataset[4][i]

        preds = model(x.unsqueeze(0), torch.ones_like(x.unsqueeze(0))).transpose(0, 1)
        pred_tree = build_tree(list(preds.data[0]), sent)
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


def batchify(dataset, cuda = False, batch_size=2, padding_idx=0):
    batches = []
    i = 0
    while i + batch_size < len(dataset[0]):
        x = dataset[0][i:i+batch_size]
        y = dataset[1][i:i+batch_size]

        max_len = 0
        for ex in x:
            if ex.shape[0] > max_len:
                max_len = ex.shape[0]
        current_x = []
        current_y = []
        current_mask_x = []
        current_mask_y = []
        for ex_x, ex_y in zip(x, y):
            mask_x = torch.ones_like(ex_x)
            mask_y = torch.ones_like(ex_y, dtype=torch.long)
            while ex_x.shape[0] < max_len:
                ex_x = torch.cat((ex_x, torch.LongTensor([padding_idx])))
                ex_y = torch.cat((ex_y, torch.FloatTensor([padding_idx])))
                mask_x = torch.cat((mask_x, torch.LongTensor([padding_idx])))
                mask_y = torch.cat((mask_y, torch.LongTensor([padding_idx])))
            current_x.append(ex_x.unsqueeze(0))
            current_y.append(ex_y.unsqueeze(0))
            current_mask_x.append(mask_x.unsqueeze(0))
            current_mask_y.append(mask_y.unsqueeze(0))
        if cuda:
            batches.append((torch.cat(current_x).cuda(), torch.cat(current_y).cuda(),
                            torch.cat(current_mask_x).cuda(), torch.cat(current_mask_y).cuda()))
        else:
            batches.append((torch.cat(current_x), torch.cat(current_y),
                            torch.cat(current_mask_x), torch.cat(current_mask_y)))
        i += batch_size

    return batches


def train_fct(train_data, valid_data, vocab, cuda=False,  nemb=100, nhid=300, epochs=300):
    model = Parser(nemb, nhid, len(vocab))
    optimizer = optim.Adam(model.parameters())
    batchify(train_data)
    train = batchify(train_data, cuda = cuda)
    print(len(train))
    
    for epoch in range(epochs):
        model.train()
        av_loss = 0.
        shuffle(train)
        for (x, y, mask_x, mask_y) in train:
            optimizer.zero_grad()
            preds = model(x, mask_x, cuda)
            loss = ranking_loss(preds.transpose(0, 1), y, mask_y)
            av_loss += loss
            loss.backward()
            optimizer.step()
        av_loss /= len(train)
        
        f1 = eval_fct(model, train_data)
        
        print('Epoch: ' + str(epoch))
        print('Loss: ' + str(av_loss.data))
        print('F1: ' + str(f1))
    return None

is_cuda = False
gpu_device = 0
if __name__ == '__main__':
    '''
       For now, the first argument is the path to the data.
       # TODO: use a reasonable argument parser!
    '''

    if torch.cuda.is_available():
        print("You are not using CUDA.")
    else:
        is_cuda = True
        torch.cuda.set_device(gpu_device)
        print("You are using CUDA.")

    train_data, valid_data, test_data = data_loader.main(sys.argv[1])
    train_fct(train_data, valid_data, valid_data[-1], is_cuda)