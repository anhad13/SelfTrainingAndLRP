import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class Parser(nn.Module):

    def __init__(self, nemb, nhid, nvoc, dropout=0.3, dropout_emb=0.1):
        super(Parser, self).__init__()
        
        self.nvoc = nvoc
        self.nemb = nemb
        self.nhid = nhid
        self.embed = nn.Embedding(self.nvoc, self.nemb, padding_idx=0)
        self.dropout_emb = dropout_emb
        
        self.lstm1 = nn.LSTM(self.nemb, self.nhid, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(2*self.nhid, self.nhid, 2)
        self.lstm2 = nn.LSTM(self.nhid, self.nhid, num_layers=1, bidirectional=True)
        self.ff1 = nn.Linear(self.nhid*2, self.nhid)
        self.ff2 = nn.Linear(self.nhid, 1)
    
        self.tanh = nn.Tanh()

    
    def forward(self, input, mask, cuda):
        # emb: seq_len, emb_size
        emb = self.embed(input)
        if self.training:
            if not cuda:
                emb_drop_mask = (torch.FloatTensor(input.shape).uniform_() > self.dropout_emb).float()  # TODO: make parameter!
            else:
                emb_drop_mask = (torch.FloatTensor(input.shape).cuda().uniform_() > self.dropout_emb).float()
            emb = torch.mul(emb, emb_drop_mask.unsqueeze(2))

        packed_sequence = pack_padded_sequence(emb, mask.data.sum(dim=1), batch_first=True, enforce_sorted=False)
        
        # lstm1_out: seq_len, batch, num_directions * hidden_size
        lstm1_out, _ = self.lstm1(packed_sequence)
        lstm1_out, _ = pad_packed_sequence(lstm1_out, batch_first=True)
        lstm1_out = lstm1_out.transpose(1,2)  #.transpose(0,1)
        lstm1_out = self.dropout(lstm1_out)
        # conv_out: bsz, hidden, seq_len-1
        conv_out = F.relu(self.conv1(lstm1_out))
        conv_out = conv_out.transpose(1,2)  #.transpose(0,1)
        conv_out = self.dropout(conv_out)
        short_mask = mask.sum(dim=1) - torch.ones_like(mask.sum(dim=1))
        
        packed_sequence = pack_padded_sequence(conv_out, short_mask, batch_first=True, enforce_sorted=False)
        lstm2_out, _ = self.lstm2(packed_sequence)
        lstm2_out, _ = pad_packed_sequence(lstm2_out, batch_first=True)
        lstm2_out = lstm2_out.transpose(0,1)
        lstm2_out = self.dropout(lstm2_out[1:-1])
        ff1_out = self.ff1(lstm2_out)
        ff1_out = self.dropout(self.tanh(ff1_out))
        ff2_out = self.ff2(ff1_out)
        
        # ff2_out: batch_size, seq_len-1, 2
        return ff2_out.squeeze(2)
