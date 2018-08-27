# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""

import torch
import torchvision.models as models


class NopModule(torch.nn.Module):
    def __init__(self):
        super(NopModule, self).__init__()

    def forward(self, input_):
        return input_


def get_cnn(device=torch.device('cpu')):

    inception = models.inception_v3(pretrained=True)  # pretrain on GoogleNet
    inception.fc = NopModule()
    inception = inception.to(device)
    inception.eval()

    return inception
