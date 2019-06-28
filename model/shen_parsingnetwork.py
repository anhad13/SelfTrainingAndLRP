'''
    This is mostly copied from https://github.com/yikangshen/PRPN/blob/master/ParsingNetwork.py
'''

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ShenParsingNetwork(nn.Module):
    def __init__(self, ninp, nhid, nslots=5, nlookback=1, resolution=0.1, dropout=0.4, hard=False):
        super(ShenParsingNetwork, self).__init__()
        
        self.nhid = nhid
        self.ninp = ninp
        self.nslots = nslots
        self.nlookback = nlookback
        self.resolution = resolution
        self.hard = hard
        
        self.drop = nn.Dropout(dropout)
        
        # Attention layers
        self.gate = nn.Sequential(nn.Dropout(dropout),
                                  nn.Conv1d(ninp, nhid, (nlookback + 1)),
                                  nn.BatchNorm1d(nhid),
                                  nn.ReLU(),
                                  nn.Dropout(dropout))
            
        self.gate2 = nn.Sequential(nn.Conv1d(nhid, 2, 1, groups=2),
                                   nn.Sigmoid())
    
        self.dist_ff = nn.Linear(self.nhid, 1)
        self.distances = None
        

    
    
    def forward(self, emb, parser_state):
        emb_last, cum_gate = parser_state
        ntimestep = emb.size(0)
        
        emb_last = torch.cat([emb_last, emb], dim=0)
        emb = emb_last.transpose(0, 1).transpose(1, 2)  # bsz, ninp, ntimestep + nlookback
        
        pre_gates = self.gate(emb)  # bsz, 2, ntimestep
        gates = self.gate2(pre_gates)
        
        self.distances = self.dist_ff(pre_gates.transpose(1, 2)).squeeze(2)
        
        gate = gates[:, 0, :]
        gate_next = gates[:, 1, :]
        cum_gate = torch.cat([cum_gate, gate], dim=1)
        gate_hat = torch.stack([cum_gate[:, i:i + ntimestep] for i in range(self.nslots, 0, -1)],
                               dim=2)  # bsz, ntimestep, nslots
            
        if self.hard:
            memory_gate = (F.hardtanh((gate[:, :, None] - gate_hat) / self.resolution * 2 + 1) + 1) / 2
        else:
            memory_gate = F.sigmoid(
                (gate[:, :, None] - gate_hat) / self.resolution * 10 + 5)  # bsz, ntimestep, nslots
        memory_gate = torch.cumprod(memory_gate, dim=2)  # bsz, ntimestep, nlookback+1
        memory_gate = torch.unbind(memory_gate, dim=1)
                                       
        if self.hard:
            memory_gate_next = (F.hardtanh((gate_next[:, :, None] - gate_hat) / self.resolution * 2 + 1) + 1) / 2
        else:
            memory_gate_next = F.sigmoid(
                (gate_next[:, :, None] - gate_hat) / self.resolution * 10 + 5)  # bsz, ntimestep, nslots
        memory_gate_next = torch.cumprod(memory_gate_next, dim=2)  # bsz, ntimestep, nlookback+1
        memory_gate_next = torch.unbind(memory_gate_next, dim=1)
                                                       
        return (memory_gate, memory_gate_next), gate, (emb_last[-self.nlookback:], cum_gate[:, -self.nslots:])

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        self.ones = Variable(weight.new(bsz, 1).zero_() + 1)
        return Variable(weight.new(self.nlookback, bsz, self.ninp).zero_()), \
               Variable(weight.new(bsz, self.nslots).zero_() + numpy.inf)
