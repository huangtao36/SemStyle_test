# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""
import torch
import torch.nn as nn


def build_model(enc_vocab_size, dec_vocab_size, hid_size=512,
                loaded_state=None, device=torch.device('cpu')):

    enc = Encoder(enc_vocab_size, hid_size)
    dec = DecoderAttn(dec_vocab_size, hid_size, dec_vocab_size)

    if loaded_state is not None:
        enc.load_state_dict(loaded_state['enc'])
        dec.load_state_dict(loaded_state['dec'])

    enc = enc.to(device)
    dec = dec.to(device)

    return enc, dec


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        assert hidden_size % 2 == 0

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.hidden_init_tensor = torch.zeros(2, 1, int(self.hidden_size / 2),
                                              requires_grad=True)
        nn.init.normal_(self.hidden_init_tensor, mean=0, std=0.05)
        self.hidden_init = torch.nn.Parameter(self.hidden_init_tensor,
                                              requires_grad=True)

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, int(hidden_size / 2), batch_first=True,
                          bidirectional=True)
        self.gru_out_drop = nn.Dropout(0.2)
        self.gru_hid_drop = nn.Dropout(0.3)

    def forward(self, input_, hidden, lengths):
        emb = self.emb_drop(self.embedding(input_))
        pp = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths,
                                                     batch_first=True)
        out, hidden = self.gru(pp, hidden)
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        out = self.gru_out_drop(out)
        hidden = self.gru_hid_drop(hidden)
        return out, hidden

    def init_hidden(self, bs):
        return self.hidden_init.expand(2, bs,
                                       int(self.hidden_size/2)).contiguous()


class DecoderAttn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderAttn, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.2)
        self.mlp = nn.Linear(hidden_size * 2, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

        self.att_mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_softmax = nn.Softmax(dim=2)

    def forward(self, input_, hidden, encoder_outs):
        emb = self.embedding(input_)
        out, hidden = self.gru(self.emb_drop(emb), hidden)

        out_proj = self.att_mlp(out)
        enc_out_perm = encoder_outs.permute(0, 2, 1)
        e_exp = torch.bmm(out_proj, enc_out_perm)
        attn = self.attn_softmax(e_exp)

        ctx = torch.bmm(attn, encoder_outs)

        full_ctx = torch.cat([self.gru_drop(out), ctx], dim=2)

        out = self.mlp(full_ctx)
        out = self.logsoftmax(out)

        return out, hidden, attn
