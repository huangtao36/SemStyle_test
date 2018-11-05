# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""

import torch
from utils_text import untokenize

from model.model_pretrain import get_cnn
from model.img2text_net import build_model


def setup_test(path, device=torch.device('cpu')):
    cnn = get_cnn(device=device)
    if torch.cuda.is_available():
        loaded_state = torch.load(path, map_location=device)
    else:
        loaded_state = torch.load(path, map_location='cpu')

    dec_vocab_size = len(loaded_state['dec_idx_to_word'])  # 10,004
    enc, dec = build_model(dec_vocab_size,
                           loaded_state=loaded_state,
                           device=device)

    return {'cnn': cnn, 'enc': enc, 'dec': dec,
            'loaded_state': loaded_state}


def generate(enc, dec, feats, len_=20, device=torch.device('cpu')):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        hid_enc = enc(feats).unsqueeze(0)      # [1, batch, 512]

        # run the decoder step by step
        dec_tensor = torch.zeros(feats.shape[0], len_ + 1, dtype=torch.long)
        dec_tensor = dec_tensor.to(device)
        last_enc = hid_enc
        for i in range(len_):
            out_dec, hid_dec = dec.forward(dec_tensor[:, i].unsqueeze(1),
                                           last_enc)
            chosen = torch.argmax(out_dec[:, 0], dim=1)
            dec_tensor[:, i + 1] = chosen
            last_enc = hid_dec

    return dec_tensor.data.cpu().numpy()


def img2text(input_, model_path, device=torch.device('cpu')):

    setup_data = setup_test(path=model_path, device=device)

    enc = setup_data['enc']
    dec = setup_data['dec']
    cnn = setup_data['cnn']
    loaded_state = setup_data['loaded_state']

    with torch.no_grad():
        batch_feats_tensor = cnn(input_)  # [batch, 2048]

    dec_tensor = generate(enc, dec, batch_feats_tensor, device=device)
    # [batch, 21]

    untok = []  # save the word
    for i in range(dec_tensor.shape[0]):
        untok.append(untokenize(dec_tensor[i],
                                loaded_state['dec_idx_to_word'],
                                to_text=False))

    return untok
