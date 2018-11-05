# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""
import os
import torch
import argparse
from PIL import Image
import cv2
import torchvision.transforms as transforms
from img2text_process import img2text
from seq2seq_process import seq2seq

gpu_id = 0

project_path = os.path.abspath(os.path.join(os.getcwd()))
model_path = os.path.join(project_path, "pretrain_param/")

test_model_fname = os.path.join(model_path, "img_to_txt_state.tar")
seq_to_seq_test_model_fname = os.path.join(model_path, "seq_to_txt_state.tar")

ROM_STYLE = "ROMANCETOKEN"
COCO_STYLE = "MSCOCOTOKEN"


class SemStyle():
    def __init__(self, global_args):

        self.args = global_args
        self.transform = set_transform()

    def process(self, im_):
        input_ = self.transform(im_).unsqueeze(0)
        input_ = input_.to(self.args.device)
        untok = img2text(input_,
                         model_path=test_model_fname,
                         device=self.args.device)

        text = seq2seq(untok,
                       model_path=seq_to_seq_test_model_fname,
                       batch_size=1,
                       style=ROM_STYLE,
                       device=self.args.device)

        return text


def set_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize,
    ])

    return trans


if __name__ == '__main__':
    DEVICE = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', default='', type=str,
                           help='device to run')

    g_args = argparser.parse_args()
    if g_args.device == '':
        g_args.device = DEVICE
    else:
        g_args.device = torch.device(DEVICE)

    im = Image.open('./123.jpg')
    sem_style = SemStyle(g_args)
    text = sem_style.process(im)
    print(text)
