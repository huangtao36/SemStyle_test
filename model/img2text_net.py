# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""
import torch
import torch.nn as nn


def build_model(dec_vocab_size, img_feat_size=2048, hid_size=512,
                loaded_state=None,  device=torch.device('cpu')):

    enc = ImgEmb(img_feat_size, hid_size)
    dec = Decoder(dec_vocab_size, hid_size, dec_vocab_size)
    if loaded_state is not None:
        enc.load_state_dict(loaded_state['enc'])
        dec.load_state_dict(loaded_state['dec'])

    enc = enc.to(device)
    dec = dec.to(device)

    return enc, dec


class ImgEmb(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImgEmb, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.mlp = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input_):
        res = self.relu(self.mlp(input_))
        return res


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.5)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.5)
        self.mlp = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, input_, hidden_in):
        emb = self.embedding(input_)
        out, hidden = self.gru(self.emb_drop(emb), hidden_in)
        out = self.mlp(self.gru_drop(out))
        out = self.logsoftmax(out)
        return out, hidden
