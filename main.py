import sys
import argparse
from random import shuffle
import numpy
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
from utils import data_loader
from model.parser import Parser
from model.prpn import PRPN
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


def build_tree_prpn(depth, sen):
    assert len(depth) == len(sen)
    
    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree_prpn(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree_prpn(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def eval_fct(model, dataset, use_prpn, cuda=False):
    model.eval()
    prec_list = []
    reca_list = []
    f1_list = []
    for i in range(len(dataset[0])):
        #for i in range(1):
        x = dataset[0][i]
        y = dataset[1][i]
        if cuda:
            x = x.cuda()
            y = y.cuda()
        gold_brackets = dataset[3][i]
        sent = dataset[4][i]

        if use_prpn:
            x = x.unsqueeze(1)
            hidden = model.init_hidden(1)
            _, hidden = model(x, hidden)
            gates = model.gates.squeeze(0).unsqueeze(1)
            preds = gates[1:-1]
            pred_tree = build_tree_prpn(list(preds.data), sent[1:-1])
        else:
            preds = model(x.unsqueeze(0), torch.ones_like(x.unsqueeze(0)), cuda).transpose(0, 1)
            pred_tree = build_tree(list(preds.data[0]), sent[1:-1])
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


def batchify(dataset, batch_size, use_prpn, cuda = False, padding_idx=0):
    batches = []
    i = 0
    while i + batch_size <= len(dataset[0]):
        x = dataset[0][i:i+batch_size]
        if use_prpn:
            y = dataset[5][i:i+batch_size]  # gates
        else:
            y = dataset[1][i:i+batch_size]  # distances

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


def LM_criterion(input, targets, targets_mask, ntokens):
    targets_mask = targets_mask.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    input = input.view(-1, ntokens)
    input = F.log_softmax(input, dim=-1)
    loss = torch.gather(input, 1, targets[:, None]).view(-1)
    loss = (-loss * targets_mask.float()).sum() / targets_mask.sum()
    return loss


def train_fct(train_data, valid_data, vocab, use_prpn, cuda=False,  nemb=100, nhid=300, epochs=300, batch_size=1,
              alpha=1., beta=0.):
    if use_prpn:
        print('Using PRPN.')
        model = PRPN(len(vocab), nemb, nhid, 2, 15, 5, 0.1, 0.2, 0.2, 0.0, False, False, 0)
    else:
        print('Using supervised parser.')
        model = Parser(nemb, nhid, len(vocab))
    optimizer = optim.Adam(model.parameters())
    train = batchify(train_data, batch_size, use_prpn, cuda = cuda)
    print('Number of training batches: ' + str(len(train)))
    if cuda:
        model.cuda()    
    for epoch in range(epochs):
        model.train()
        count = 0
        epoch_start_time = time.time()
        av_loss = 0.
        shuffle(train)
        for (x, y, mask_x, mask_y) in train:
            optimizer.zero_grad()
            if use_prpn:
                hidden = model.init_hidden(batch_size)
                output, _ = model(x.transpose(1, 0), hidden)
                zeros = torch.zeros((mask_x.shape[0],)).unsqueeze(0).long()
                gates = model.gates.transpose(1, 0)[1:-1].transpose(1, 0)
                loss1 = ranking_loss(gates, y, mask_y)
                loss2 = LM_criterion(output, torch.cat([x.transpose(1, 0)[1:], zeros], dim=0),
                                    torch.cat([mask_x.transpose(1, 0)[1:], zeros], dim=0), len(vocab))
                loss = alpha * loss1 + beta * loss2
            else:
                preds = model(x, mask_x, cuda)
                loss = ranking_loss(preds.transpose(0, 1), y, mask_y)
            av_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()
            if count % 100 == 0:
                print("Epoch: "+str(epoch)+" -- batch: "+str(count))
            count+=1
        av_loss /= len(train)
        print("Training time for epoch in sec: ", (time.time()-epoch_start_time))
        f1 = eval_fct(model, train_data, use_prpn, cuda)
        
        print('End of epoch ' + str(epoch))
        print('Loss: ' + str(av_loss.data))
        print('F1: ' + str(f1))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing and grammar induction')
    parser.add_argument('--data', type=str, default='data/', help='location of the data corpus')
    parser.add_argument('--PRPN', action='store_true',
                        help='use PRPN; otherwise, use the parser')
    args = parser.parse_args()
    
    is_cuda = False
    gpu_device = 0
    if not torch.cuda.is_available():
        print("You are not using CUDA.")
    else:
        is_cuda = True
        torch.cuda.set_device(gpu_device)
        print("You are using CUDA.")

    train_data, valid_data, test_data = data_loader.main(args.data)
    train_fct(train_data, valid_data, valid_data[-1], args.PRPN, is_cuda)
