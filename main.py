import sys
import os
import argparse
from random import shuffle
import numpy
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
from utils import data_loader, ctb_data, german_data, arabic_data, ctb_data_wkp
from model.parser import Parser
from model.prpn import PRPN
from utils.data_loader import build_tree, get_brackets, build_tree_labelled, get_pred_labelled_bracketed
import torch.nn as nn
import pickle
import math
from os import listdir
from os.path import isfile, join, isdir
from utils.compute_f1 import compute_f1
# do pip install PYEVALB before running this.
from PYEVALB import scorer
from PYEVALB import parser

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


def eval_fct(model, dataset, use_prpn, parse_with_gates, cuda=False, output_file=None):
    model.eval()
    prec_list = []
    reca_list = []
    f1_list = []
    outf = []
    label_map = dataset[-2]
    label_targets = dataset[7]
    #first reverse label map
    rev_label_map = {}
    accuracy_map = {}
    do_labelled_f1 = False
    net_acc = []
    labelled_f1 = []
    for x in label_map.keys():
        accuracy_map[x] = []
        rev_label_map[label_map[x]] = x
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
            if parse_with_gates:  # "normal" way of parsing with PRPN
                gates = model.gates.squeeze(0).unsqueeze(1)
                preds = gates[1:-1]
                pred_tree = build_tree_prpn(list(preds.data), sent[1:-1])
            else:  # parse using supervised distances
                do_labelled_f1 = True
                preds = model.distances.transpose(1, 0)[2:-1].squeeze(0)
                pred_tree = build_tree(list(preds.data), sent[1:-1])
                pred_tree_labelled = build_tree_labelled(list(preds.data), sent[1:-1], list(model.label_out.transpose(0,1)[2:-1].squeeze(1).argmax(1)), rev_label_map)
                predicted_nonleafs = list(model.label_out.transpose(0,1)[2:-1].squeeze(1).argmax(1))
                predicted_leafs = list(model.leaf_label_out.squeeze(1).argmax(1))
                #gold_leafs, gold_nonleafs, gold_dists,
                l_f1 = compute_f1(dataset[4][i], dataset[10][i],predicted_leafs, predicted_nonleafs, list(preds.data), list(dataset[11][i]),list(dataset[9][i]),list(dataset[7][i]) ,list(dataset[1][i]),rev_label_map)
                label_brackets = get_pred_labelled_bracketed(pred_tree_labelled)[0]
                do_labelled_f1 = True
        else:
            preds = model(x.unsqueeze(0), torch.ones_like(x.unsqueeze(0)), cuda).transpose(0, 1)
            pred_tree = build_tree(list(preds.data[0]), sent[1:-1])
            pred_tree_labelled = build_tree_labelled(list(preds.data[0]), sent[1:-1], list(model.label_out.transpose(0,1)[0].argmax(1)), rev_label_map)
            predicted_nonleafs = list(model.label_out.transpose(0,1)[0].argmax(1))
            predicted_leafs = list(model.leaf_label_out.squeeze(1).argmax(1))
            #gold_leafs, gold_nonleafs, gold_dists,
            l_f1 = compute_f1(dataset[4][i], dataset[10][i],predicted_leafs, predicted_nonleafs, list(preds.data[0]), list(dataset[11][i]),list(dataset[9][i]),list(dataset[7][i]) ,list(dataset[1][i]),rev_label_map)
            label_brackets = get_pred_labelled_bracketed(pred_tree_labelled)[0]
            do_labelled_f1 = True
        predicted_labels = model.label_out[0].argmax(1)[2:-1]
        #net_acc.append(torch.sum(predicted_labels ==label_targets[i]).cpu().data.item()/float(len(predicted_labels)))
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
        if do_labelled_f1:
            labelled_f1.append(l_f1)
        outf.append({'f1': f1, 'example': sent, 'pred_tree': pred_tree, 'preds': preds, 'parse_with_gates': parse_with_gates, 'gold': gold_brackets})    
    if output_file:
        f = open(output_file, "wb")
        pickle.dump(outf, f)
    # Sentence-level F1.
    if do_labelled_f1:
        print("Labelled F1", numpy.mean(labelled_f1))
        # print("Label Acc", numpy.mean(net_acc))
    return numpy.mean(f1_list)


def batchify(dataset, batch_size, use_prpn, cuda = False, padding_idx=0, training_method = "unsupervised", training_ratio = 0.5):
    #batching options = interleave, supervised, unsupervised
    batches = []
    i = 0
    while i + batch_size <= len(dataset[0]):
        x = dataset[0][i:i+batch_size]
        yg = dataset[5][i:i+batch_size]  # [5] for gates
        yd = dataset[1][i:i+batch_size]  # distances
        skip_sup = dataset[6][i:i+batch_size]  # distances
        ll = dataset[7][i:i+batch_size]
        leafl = dataset[9][i:i+batch_size]
        max_len = 0
        for ex in x:
            if ex.shape[0] > max_len:
                max_len = ex.shape[0]
        current_x = []
        current_yg = []
        current_yd = []
        current_mask_x = []
        current_mask_yd = []
        current_mask_yg = []
        current_mask_mg = []
        current_mask_md = []
        current_ll = []
        current_leafl = []
        skip_no = 0
        for ex_x, ex_yg, ex_yd, ex_ll, ex_leaf in zip(x, yg, yd, ll, leafl):
            mask_x = torch.ones_like(ex_x)
            mask_yg = torch.ones_like(ex_yg, dtype=torch.long)
            mask_yd = torch.ones_like(ex_yd, dtype=torch.long)
            repl_x = ex_x
            while ex_x.shape[0] < max_len:
                ex_x = torch.cat((ex_x, torch.LongTensor([padding_idx])))
                ex_yg = torch.cat((ex_yg, torch.FloatTensor([padding_idx])))
                ex_yd = torch.cat((ex_yd, torch.FloatTensor([padding_idx])))
                ex_ll = torch.cat((ex_ll, torch.LongTensor([padding_idx])))
                ex_leaf = torch.cat((ex_leaf, torch.LongTensor([padding_idx])))
                mask_x = torch.cat((mask_x, torch.LongTensor([padding_idx])))
                mask_yd = torch.cat((mask_yd, torch.LongTensor([padding_idx])))
                mask_yg = torch.cat((mask_yg, torch.LongTensor([padding_idx])))

            # 1 - > -1 is valid
            mask_mg = torch.cat((torch.zeros(1), torch.ones(len(repl_x[1:-1])), torch.zeros(max_len-len(repl_x[1:-1])-1)))
            # 2 -> -1
            mask_md = torch.cat((torch.zeros(2), torch.ones(len(repl_x[1:-1])-1), torch.zeros(max_len-len(repl_x[1:-1])-1)))
            for_supervision_limitg = torch.clamp(ex_yg, 0.0, 1.0).long()
            mask_yg = for_supervision_limitg * mask_yg  # setting mask_y to zero for examples without supervision
            for_supervision_limitd = torch.clamp(ex_yd, 0.0, 1.0).long()
            mask_yd = for_supervision_limitd * mask_yd  # setting mask_y to zero for examples without supervision
            current_x.append(ex_x.unsqueeze(0))
            current_yg.append(ex_yg.unsqueeze(0))
            current_yd.append(ex_yd.unsqueeze(0))
            current_mask_x.append(mask_x.unsqueeze(0))
            current_mask_yd.append(mask_yd.unsqueeze(0))
            current_mask_yg.append(mask_yg.unsqueeze(0))
            current_mask_mg.append(mask_mg.unsqueeze(0))
            current_mask_md.append(mask_md.unsqueeze(0))
            current_ll.append(ex_ll.unsqueeze(0))
            current_leafl.append(ex_leaf.unsqueeze(0))
        supervision_type = ["unsupervised"]
        if training_method == 'interleave':
            if max(skip_sup) == False: #null supervision on this:
                supervision_type = ["unsupervised"]
            else:
                supervision_type = ["supervised", "unsupervised"]
        elif training_method == 'supervised':
            supervision_type = ["supervised"]
        elif training_method == 'semisupervised':
            supervision_type = ["semisupervised"]
        for is_batch_supervised in supervision_type:
            if cuda:
                batches.append((torch.cat(current_x).cuda(), torch.cat(current_yd).cuda(), torch.cat(current_yg).cuda(),
                                torch.cat(current_mask_x).cuda(), torch.cat(current_mask_yd).cuda(), 
                                torch.cat(current_mask_yg).cuda() ,torch.cat(current_mask_mg).cuda(), torch.cat(current_mask_md).cuda(), 
                                is_batch_supervised, torch.cat(current_ll).cuda(), torch.cat(current_leafl).cuda()))
            else:
                batches.append((torch.cat(current_x), torch.cat(current_yd), torch.cat(current_yg),
                                torch.cat(current_mask_x), torch.cat(current_mask_yd), torch.cat(current_mask_yg), 
                                torch.cat(current_mask_mg), torch.cat(current_mask_md), 
                                is_batch_supervised, torch.cat(current_ll), torch.cat(current_leafl)))
        i += batch_size
    if training_ratio == 1.0 and training_method == 'interleave':
        training_method = "supervised"
    elif training_ratio == 0.0 and training_method == 'interleave':
        training_method = "unsupervised"
    if training_method == "interleave": # interleave batches based on [-1]
        supervised_batches = []
        unsupervised_batches = []
        for batch in batches:
            if batch[-1] == "supervised":
                supervised_batches.append(batch)
            else:
                unsupervised_batches.append(batch)
        batches = []
        supi = 0
        unsupi = 0
        lens = len(supervised_batches)
        lenu = len(unsupervised_batches)
        no_super = lens
        print("Init batches: no of supervised: "+str(lens)+", no of UNS: "+str(lenu))
        if lens < lenu:
            no_super = math.ceil(max(lenu*(training_ratio/(1-training_ratio)), lens))
            is_super = numpy.zeros(no_super + lenu)
            is_super[0: no_super] = 1
            print("Applying ratio: no of supervised: "+str(no_super)+", no of unsupervised: " +str(lenu))
        else:
            no_unsuper = math.ceil(max(lens * (1-training_ratio)/training_ratio, lenu))
            is_super = numpy.ones(no_unsuper + lens)
            is_super[0: no_unsuper] = 0
            print("Applying ratio: no of supervised: "+str(lens)+", no of unsupervised: " +str(no_unsuper))
        shuffle(is_super)
        for i in range(len(is_super)):
            if is_super[i] == 1:
                batches.append(supervised_batches[supi])
                supi = (supi + 1)%len(supervised_batches)
            else:
                batches.append(unsupervised_batches[unsupi])
                unsupi = (unsupi + 1)%len(unsupervised_batches)
    return batches


def LM_criterion(input, targets, targets_mask, ntokens):
    targets_mask = targets_mask.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    input = input.view(-1, ntokens)
    input = F.log_softmax(input, dim=-1)
    loss = torch.gather(input, 1, targets[:, None]).view(-1)
    loss = (-loss * targets_mask.float()).sum() / targets_mask.sum()
    return loss


def train_fct(train_data, valid_data, vocab, use_prpn, cuda=False,  nemb=100, nhid=300, epochs=300, batch_size=3,
              alpha=0., train_beta=1.0, parse_with_gates=True, save_to=None, load_from=None, eval_on='dev',
              use_orig_prpn=False, training_method='unsupervised', training_ratio=0.5, label_weight = 100.0):
    if save_to:
        if '/' in save_to:
            os.makedirs('/'.join(save_to.split('/')[:-1]), exist_ok=True)
    if use_prpn:
        info = 'Using PRPN, ' + str(train_beta) + ' of gates and ' + str(1 - train_beta) + ' of distances.'
        if alpha == 0.:
            info += 'unsupervised.'
        if parse_with_gates:
            info += '\nUsing gate values for parsing.'
        else:
            info += '\nUsing distances for parsing.'
        print(info)
        model = PRPN(len(vocab), nemb, nhid, 2, 15, 5, 0.1, 0.2, 0.2, 0.0, False, False, 0, use_orig_prpn=use_orig_prpn, nlabels= len(train_data[-2]))
    else:
        print('Using supervised parser.')
        model = Parser(nemb, nhid, len(vocab), nlabels=len(train_data[-2]))
    if load_from:
        print('Loading pretrained model from ' + load_from + '.')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(load_from, map_location=device)

    optimizer = optim.Adam(model.parameters())
    if batch_size > len(train_data[0]):
        print('Reducing batch size to ' + str(len(train_data[0])) + ' due to train set size.')
        batch_size = len(train_data[0])
    train = batchify(train_data, batch_size, use_prpn, cuda = cuda, training_method = training_method, training_ratio=training_ratio)
    print('Number of training batches: ' + str(len(train)))
    if cuda:
        model.cuda()
    max_f1 = -1
    for epoch in range(epochs):
        model.train()
        count = 0
        epoch_start_time = time.time()
        av_loss = 0.
        shuffle(train)
        nlabels = len(train_data[-2])
        for (x, yd, yg, mask_x, mask_yd, mask_yg, mask_mg, mask_md, training_method, label_l, leaf_l) in train:
            optimizer.zero_grad()
            if use_prpn:
                if training_method == "unsupervised":
                    alpha = 0.0
                elif training_method == "supervised":
                    alpha = 1.0
                hidden = model.init_hidden(batch_size)
                output, _ = model(x.transpose(1, 0), hidden)
                if cuda:
                    zeros = torch.zeros((mask_x.shape[0],)).unsqueeze(0).cuda().long()
                else:
                    zeros = torch.zeros((mask_x.shape[0],)).unsqueeze(0).long()
                gates = model.gates * mask_mg
                gates = gates.transpose(0,1)[1:-1].transpose(0,1)
                loss1g = ranking_loss(gates, yg, mask_yg)
                # multi-task training on distances
                distances = model.distances * mask_md
                distances = distances.transpose(0,1)[2:-1].transpose(0,1)
                loss1d = ranking_loss(distances, yd, mask_yd)
                label_out = model.label_out.contiguous().view(-1, nlabels)
                leaf_label_out = model.leaf_label_out.transpose(0,1).contiguous().view(-1, nlabels)
                label_l = torch.cat([torch.zeros(batch_size,2).long(),label_l, torch.zeros(batch_size,1).long()], 1)
                leaf_l = torch.cat([torch.zeros(batch_size,1).long(),leaf_l, torch.zeros(batch_size,1).long()], 1)
                leaf_loss = nn.CrossEntropyLoss(ignore_index=0)(leaf_label_out, leaf_l.contiguous().view(-1))
                loss_labels = nn.CrossEntropyLoss(ignore_index=0)(label_out, label_l.contiguous().view(-1))
                loss_labels += leaf_loss
                loss1 = loss1g * train_beta + loss1d * (1 - train_beta)
                loss2 = LM_criterion(output, torch.cat([x.transpose(1, 0)[1:], zeros], dim=0),
                                     torch.cat([mask_x.transpose(1, 0)[1:], zeros], dim=0), len(vocab))
                loss = alpha * loss1 + (1 - alpha) * loss2
                if training_method!="unsupervised":
                    loss += label_weight * loss_labels
            else:
                #straight to the tree.
                preds = model(x, mask_x, cuda)
                label_out = model.label_out.transpose(0,1).contiguous().view(-1, nlabels)
                leaf_label_out = model.leaf_label_out[1:-1].transpose(0,1).contiguous().view(-1, nlabels)
                leaf_loss = nn.CrossEntropyLoss(ignore_index=0)(leaf_label_out, leaf_l.contiguous().view(-1))
                loss_labels = nn.CrossEntropyLoss(ignore_index=0)(label_out, label_l.contiguous().view(-1))
                loss_labels += leaf_loss
                loss = ranking_loss(preds.transpose(0, 1), yd, mask_yd)
                loss += label_weight * loss_labels
            av_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()
            if count % 100 == 0:
                print("Epoch: "+str(epoch)+" -- batch: "+str(count))
            count+=1
        av_loss /= len(train)
        print("Training time for epoch in sec: ", round((time.time()-epoch_start_time), 4))
        print('End of epoch ' + str(epoch) + '. Evaluation on ' + eval_on + '.')
        if eval_on == 'train':
            f1 = eval_fct(model, train_data, use_prpn, parse_with_gates, cuda)
        elif eval_on == 'test':
            f1 = eval_fct(model, test_data, use_prpn, parse_with_gates, cuda)
        else:
            f1 = eval_fct(model, valid_data, use_prpn, parse_with_gates, cuda)
        if save_to:
            print('Storing current model...')
            torch.save(model, save_to)
        if f1 > max_f1:
            max_f1 = f1
            if save_to:
                print('Storing new best model...')
                torch.save(model, save_to + '.best')

        print('Loss: ' + str(av_loss.data))
        print('F1: ' + str(round(f1, 6)) + ' (best: ' + str(round(max_f1, 6)) + ')')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing and grammar induction')
    parser.add_argument('--data', type=str, default='data/', help='location of the data corpus')
    parser.add_argument('--save', type=str, default=None, help='path where model will be stored')
    parser.add_argument('--load', type=str, default=None, help='path to load a model from')
    parser.add_argument('--PRPN', action='store_true',
                        help='use PRPN; otherwise, use the parser')
    parser.add_argument('--shen', action='store_true',
                        help='use parsing network from Shen et al.')
    parser.add_argument('--eval_on', type=str, default='dev', help='[train|dev|test]')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='0: train distances, 1: train gates')
    parser.add_argument('--parse_with_distances', action='store_true',
                        help='use distances to build the parse tree for eval (instead of gate values)')
    parser.add_argument('--alpha', type=float, default=0.,
                        help='weight of the SUPERVISED loss for PRPN; 0. means UNSUPERVISED (default)')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--supervision_limit', type=int, default=-1, help='amount examples with supervision')
    parser.add_argument('--eval_only', action='store_true', help='flag for eval without training')
    parser.add_argument('--vocabulary', type=str, default=None, help='vocab pickled file path')
    parser.add_argument('--dump_vocabulary', action='store_true', help='flag for dumping vocab.')
    parser.add_argument('--train_from_pickle',type=str,default= None, help='loading training data from pickled file.')
    parser.add_argument('--training_method', type=str, default='unsupervised', help='unsupervised/supervised/interleave/semisupervised')
    parser.add_argument('--training_ratio', type=float, default=0.5,
                        help='1: all batches SUP, 0: all UNSUP')
    parser.add_argument('--bagging', action='store_true', help='if using pickled random forest data.')
    parser.add_argument('--treebank', type= str, default='ptb', help='ptb/ctb')
    parser.add_argument('--semisupervised', action='store_true', help='do both supervi and unsupervi/useful for supervision limit types')
    parser.add_argument('--force_binarize', action='store_true', help='force for binary comparison for F1 calc')
    parser.add_argument('--nhid', type= int, default = 300, help='hidden dims')
    parser.add_argument('--nemb', type= int, default = 100, help= 'emb dimension')
    parser.add_argument('--nlookback', type= int, default = 1, help= 'lookback for PRPN')
    parser.add_argument('--label_weight', type= float, default = 100.0, help= 'label weight ')
    
    args = parser.parse_args()
    if args.treebank == "ctb":
        print("Using chinese treebank")
        data_loader = ctb_data
    elif args.treebank == "ctb_wkp":
        print("Using chinese treebank")
        data_loader = ctb_data_wkp
    elif args.treebank == "negra":
        print("Using german (negra) corpus")
        data_loader = german_data
    elif args.treebank == "arabic":
        print("Using arabic treebank")
        data_loader = arabic_data        
    else:
        print("Using english treebank")
    is_cuda = False
    gpu_device = 0
    if args.bagging:
        print("bagging...")
    if not torch.cuda.is_available():
        print("You are not using CUDA.")
    else:
        is_cuda = True
        torch.cuda.set_device(gpu_device)
        print("You are using CUDA.")
    print("training method: " + str(args.training_method))
    if args.eval_only:
        assert args.load != None
        print("Supervision limit: " + str(args.supervision_limit))
        print('Loading pretrained model from ' + args.load + '.')
        outfile = args.load + '_output_' + str(time.time())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(args.load, map_location=device) 
        train_data, valid_data, test_data = data_loader.main(args.data, supervised_model=(not args.PRPN or args.alpha == 1.), supervision_limit=args.supervision_limit, bagging=args.bagging,semisupervised = args.semisupervised, force_binarize = args.force_binarize) 
        if args.eval_on == 'train':
            f1 = eval_fct(model, train_data, args.PRPN, (not args.parse_with_distances), is_cuda, outfile)
        elif args.eval_on == 'test':
            f1 = eval_fct(model, test_data, args.PRPN, (not args.parse_with_distances), is_cuda, outfile)
        else:
            f1 = eval_fct(model, valid_data, args.PRPN, (not args.parse_with_distances), is_cuda, outfile)
        print("F1: " + str(f1))
        exit()
    if args.vocabulary:
        vocab =  pickle.load(open(args.vocabulary, "rb"))
        train_data, valid_data, test_data = data_loader.main(args.data, vocabulary = vocab, supervision_limit=args.supervision_limit, supervised_model=(not args.PRPN or args.alpha == 1.), pickled_file_path =args.train_from_pickle, bagging=args.bagging,semisupervised = args.semisupervised, force_binarize = args.force_binarize)
    else:
        train_data, valid_data, test_data = data_loader.main(args.data, supervision_limit=args.supervision_limit, supervised_model=(not args.PRPN or args.alpha == 1.), pickled_file_path =args.train_from_pickle, bagging=args.bagging ,semisupervised = args.semisupervised, force_binarize = args.force_binarize)        
    if args.dump_vocabulary:
        pickle.dump(valid_data[-1], open("dict_ctb.pkl","wb"))
        print("Saving Vocab to file.")
    train_fct(train_data, valid_data, valid_data[-1], args.PRPN, is_cuda, alpha=args.alpha,
              train_beta = args.beta, parse_with_gates=(not args.parse_with_distances),
              save_to=args.save, load_from=args.load, eval_on=args.eval_on, batch_size=args.batch, epochs=args.epochs,             
              use_orig_prpn=args.shen, training_method=args.training_method, training_ratio=args.training_ratio, nhid=args.nhid, nemb=args.nemb, label_weight = args.label_weight)
