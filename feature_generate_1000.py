import os
import re
import cv2
import torch
import time
import torch.utils.data as data
import torchvision.transforms as transforms
from img2text_process import img2text
from seq2seq_process import seq2seq
from PIL import Image
import numpy as np

# ==========Set global variables =========== #
GPU_ID = 0

model_path = "./pretrain_param/"
test_model_fname = os.path.join(model_path, "img_to_txt_state.tar")
seq_to_seq_test_model_fname = os.path.join(model_path, "seq_to_txt_state.tar")

ROM_STYLE = "ROMANCETOKEN"
COCO_STYLE = "MSCOCOTOKEN"

device = torch.device('cuda:{}'.format(GPU_ID)) \
    if torch.cuda.is_available() else torch.device('cpu')
    
project_par_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
video_dir = os.path.join(project_par_path, 'dataset_source/Youtube/Video')


need_file_list = []
for target in os.listdir(video_dir):
    if '.avi' in target:
        need_file_list.append(target)
    else: continue

print("File num: ", len(need_file_list))

def set_transform():
    normalize_ = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize_,
    ])

    return trans

transform = set_transform()

def save2txt(data, path):
    with open(path, "a") as log_file:
        log_file.write('%s\n' % data)


for i, file in enumerate(need_file_list):
    print("already processed: ", i)
    
    video_file = os.path.join(video_dir, file)
    
    video_capture = cv2.VideoCapture(video_file)
    frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)  # 帧数
    print("Total number of frames: ", frames)
    
    message = file + ', Total number of frames: ' + str(int(frames))
    txt_savepath = os.path.join('./video_attn_txt', '%s.txt'%file.split('.')[0])
    save2txt(message, txt_savepath)
    
    feature_list = []
    for n in range(int(frames)):
        print(n)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, n)
        bool_, img = video_capture.read()
        if bool_ is True:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            input_ = transform(pil_img).unsqueeze(0).to(device)
            
            # img2text
            untok = img2text(input_, 
                             model_path=test_model_fname, 
                             device=device)
            # text2text
            text, mlp_out = seq2seq(untok, 
                                    model_path=seq_to_seq_test_model_fname,
                                    batch_size=1, 
                                    style=ROM_STYLE, 
                                    device=device)
            
            message = str(n) + ': ' + str(text)
            save2txt(message, txt_savepath)
            
            mlp_out = mlp_out.squeeze(1).cpu()
            feature_list.append(mlp_out)

        if (n + 1) % 1000 == 0:
            features = torch.stack(feature_list)
            file_name = file.split('.')[0] + '_%d.c'%(n // 1000)
            torch.save(features, './video_attn_features/%s'%file_name)
            feature_list = []

    features = torch.stack(feature_list)
    file_name = file.split('.')[0] + '_last.c'
    torch.save(features, './video_attn_features/%s'%file_name)

    print('-----------------------------------------------------------------')
