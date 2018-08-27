# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""
import os
import torch

from img2text_process import img2text
from seq2seq_process import seq2seq
from dataload.dataload import get_image_reader


# ==========Set global variables =========== #
gpu_id = 0
BATCH_SIZE = 3
model_path = "./pretrain_param/"
test_model_fname = os.path.join(model_path, "img_to_txt_state.tar")
seq_to_seq_test_model_fname = os.path.join(model_path, "seq_to_txt_state.tar")
ROM_STYLE = "ROMANCETOKEN"
COCO_STYLE = "MSCOCOTOKEN"
test_folder = "./test_img/"

device = torch.device('cuda:{}'.format(gpu_id)) \
    if torch.cuda.is_available() else torch.device('cpu')
# ========================================== #


def process():

    img_reader = get_image_reader(test_folder, BATCH_SIZE)

    all_text = []
    for input_, text_data in img_reader:
        """
        input: [batch, 3, 299, 299]
        text_data: list of img path(include file), 
        """
        input_ = input_.to(device)
        untok = img2text(input_, model_path=test_model_fname, device=device)
        text = seq2seq(untok, model_path=seq_to_seq_test_model_fname,
                       batch_size=BATCH_SIZE, style=ROM_STYLE, device=device)

        all_text.extend(text)

    return all_text


if __name__ == '__main__':
    all_text = process()
    print(all_text)
