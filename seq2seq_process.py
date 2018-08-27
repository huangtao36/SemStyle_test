# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""

import torch
import numpy as np
from utils_text import tokenize_text, untokenize, pad_text, Toks
from model.seq2seq_net import build_model


def setup_test(path, device=torch.device('cpu')):
    if torch.cuda.is_available():
        loaded_state = torch.load(path, map_location=device)
    else:
        loaded_state = torch.load(path, map_location='cpu')

    enc_idx_to_word = loaded_state['enc_idx_to_word']
    enc_word_to_idx = loaded_state['enc_word_to_idx']
    enc_vocab_size = len(enc_idx_to_word)

    dec_idx_to_word = loaded_state['dec_idx_to_word']
    dec_word_to_idx = loaded_state['dec_word_to_idx']
    dec_vocab_size = len(dec_idx_to_word)

    enc, dec = build_model(enc_vocab_size,
                           dec_vocab_size,
                           loaded_state=loaded_state,
                           device=device)

    return {'enc': enc, 'dec': dec,
            'enc_idx_to_word': enc_idx_to_word,
            'enc_word_to_idx': enc_word_to_idx,
            'enc_vocab_size': enc_vocab_size,
            'dec_idx_to_word': dec_idx_to_word,
            'dec_word_to_idx': dec_word_to_idx,
            'dec_vocab_size': dec_vocab_size}


def make_packpadded(s, e, enc_padded_text, device=torch.device('cpu')):
    text = enc_padded_text[s:e]
    lengths = np.count_nonzero(text, axis=1)
    order = np.argsort(-lengths)
    new_text = text[order]
    new_enc = torch.tensor(new_text)
    new_enc = new_enc.to(device)

    leng = torch.tensor(lengths[order])
    leng.to(device)
    return order, new_enc, leng


def generate(enc, dec, enc_padded_text, len_=20, device=torch.device('cpu')):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        # run the encoder
        order, enc_pp, enc_lengths = make_packpadded(0,
                                                     enc_padded_text.shape[0],
                                                     enc_padded_text,
                                                     device=device)
        hid = enc.init_hidden(enc_padded_text.shape[0])
        out_enc, hid_enc = enc(enc_pp, hid, enc_lengths)

        hid_enc = torch.cat([hid_enc[0, :, :], hid_enc[1, :, :]],
                            dim=1).unsqueeze(0)

        # run the decoder step by step
        dec_tensor = torch.ones((enc_padded_text.shape[0]),
                                len_ + 1,
                                dtype=torch.long) * Toks.SOS
        dec_tensor = dec_tensor.to(device)
        last_enc = hid_enc
        for i in range(len_):
            out_dec, hid_dec, attn = dec.forward(dec_tensor[:, i].unsqueeze(1),
                                                 last_enc,
                                                 out_enc)
            out_dec[:, 0, Toks.UNK] = -np.inf  # ignore unknowns
            chosen = torch.argmax(out_dec[:, 0], dim=1)
            dec_tensor[:, i + 1] = chosen
            last_enc = hid_dec

    return dec_tensor.data.cpu().numpy()[np.argsort(order)]


def seq2seq(input_seqs, model_path, batch_size,
            style="MSCOCOTOKEN", device=torch.device('cpu')):

    setup_data = setup_test(path=model_path, device=device)

    input_rems_text = input_seqs
    slen = len(input_seqs)
    for i in range(slen):
        input_rems_text[i].append(style)

    _, _, enc_tok_text, _ = tokenize_text(input_rems_text,
                                          idx_to_word=setup_data[
                                              'enc_idx_to_word'],
                                          word_to_idx=setup_data[
                                              'enc_word_to_idx'])
    enc_padded_text = pad_text(enc_tok_text)

    dlen = enc_padded_text.shape[0]
    num_batch = int(dlen / batch_size)
    if dlen % batch_size != 0:
        num_batch += 1
    res = []
    for i in range(num_batch):
        dec_tensor = generate(setup_data['enc'],
                              setup_data['dec'],
                              enc_padded_text[
                              i * batch_size:(i + 1) * batch_size],
                              device=device)
        res.append(dec_tensor)

    text = []
    res = np.concatenate(res, axis=0)
    for row in res:
        utok = untokenize(row, setup_data['dec_idx_to_word'], to_text=True)
        text.append(utok)

    return text
