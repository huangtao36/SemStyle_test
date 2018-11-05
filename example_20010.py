import argparse
from PIL import Image
import torch
import cv2
import numpy as np
from utils.svr import app
from utils.yolov3.yolo_mod import YoloMod
from utils.cganimstyler.cgan_mod import CGANMod
from utils.pose.pose_mod import PoseEstimate
form utils.sem_style.semStyle_mod import SemStyle
from utils.unit.unit_mod import UNITMod
from utils.transformer.neuralstyle.transformer_mod import TransformerMod
from utils.mnist.mnist_mod import MNISTMod
DEBUG = False
DEVICE = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

#### Dealing with parameters ####
argparser = argparse.ArgumentParser()
argparser.add_argument('--device', default='', type=str, help='device to run')
g_args = argparser.parse_args()

if g_args.device == '':
    g_args.device = DEVICE
else:
    g_args.device = torch.device(DEVICE)

#### Global module objects and visualiser objects ####

# each module accepts PIL image and output PIL image
mods = dict(yolo=YoloMod(g_args),
            cgan=CGANMod(g_args),
            pose=PoseEstimate(g_args),
            img2text=SemStyle(g_args),
            unit=UNITMod(g_args),
            transformer=TransformerMod(g_args),
            mnist=MNISTMod())
#mods = dict(pose=PoseEstimate(g_args))

def img_preprocess(img, dtype='PIL', mode='RGB'):
    """
    :param img: input, it is PIL Image
    :param dtype: 'PIL' or 'Arr'
    :param mode: 'RGB', 'BGR' or 'GRAY', Note that PIL type must be 'RGB' or 'GRAY'
    """
    if dtype == 'PIL':
        image = img.convert('RGB') if mode == 'RGB' else img.convert('L')

    elif dtype == 'Arr':
        im = np.asarray(img)
        if mode == 'BGR':
            image = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        elif mode == 'GRAY':
            image = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        else:
            image = im
    else:
        image = img

    return image

def img_uniform(img, dtype='PIL', mode='RGB'):
    """
    Unified output cv2 type format, 
    :param img: input
    :param dtype: input image type, 'PIL' or 'Arr'
    :param mode: input image mode, 'RGB'-->'BGR' , if it is 'BGR' or 'GRAY', do nothing
    """
    if dtype == 'PIL':
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
    elif dtype == 'Arr':
        
        im = np.asarray(img)
        if mode == 'RGB':
            image = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        else: 
            image = im
    else: 
        image = img
        
    return image


def impostproc(im, mode_in='', mode_out=''):
    if mode_in == mode_out:
        outim = im
    return outim


def imread(fname):
    im = np.asarray(Image.open(fname).convert('RGB'))
    if im.ndim == 3 and im.shape[2] > 3:
        im = im[:, :, :3]
    return np.ascontiguousarray(im)


def pose_proc(img):
    # im = imread(fname)
    im = img_preprocess(img, dtype='Arr', mode='BGR')
    outim = mods['pose'].process(im)
    out = img_uniform(outim, dtype='Arr', mode='BGR')
    return out

def img2text_proc(img):
    # im = imread(fname)
    im = img_preprocess(img, dtype='PIL', mode='RGB')
    text = mods['pose'].process(im)
    return text

def yolo_proc(img):
    # im = imread(fname)
    im = img_preprocess(img, dtype='Arr', mode='BGR')
    outim = mods['yolo'].process(im)
    print('yolo done')
    return outim


def cgan_proc(img):
    # print("Processing cgan {}".format(fname))
    img = img_preprocess(img, dtype='PIL', mode='RGB')
    # im = Image.open(fname).convert('RGB')
    outim = mods['cgan'].process(img)
    out = img_uniform(outim, dtype='Arr', mode='RGB')
    print("Processed ")

    return out


def gta2city(img):
    img = img_preprocess(img, dtype='PIL', mode='RGB')
    mods['unit'].process(img, 1)
    print("Processed ")


def city2gta(img):
    img = img_preprocess(img, dtype='PIL', mode='RGB')
    mods['unit'].process(img, 0)
    print("Processed ")


def starry_night(img):
    img = img_preprocess(img, dtype='PIL', mode='RGB')
    mods['transformer'].process(img, 'starry-night.pth')
    print("Processed ")


def mosaic(img):
    img = img_preprocess(img, dtype='PIL', mode='RGB')
    mods['transformer'].process(img, 'mosaic.pth')
    print("Processed ")

def udnie(img):
    img = img_preprocess(img, dtype='PIL', mode='RGB')
    mods['transformer'].process(img, 'udnie.pth')
    print("Processed ")

def candy(img):
    img = img_preprocess(img, dtype='PIL', mode='RGB')
    mods['transformer'].process(img, 'candy.pth')
    print("Processed ")

def mnist(img):
    img = img_preprocess(img, dtype='PIL', mode='L')
    img=mods['mnist'].process(img)
    print("Processed ")
    return img

app.upload_callbacks = dict(detect=yolo_proc,
                            style_vangogh=cgan_proc,
                            skeleton=pose_proc,
                            gta2city=gta2city,
                            city2gta=city2gta,
                            starry_night=starry_night,
                            mosaic=mosaic,
                            udnie=udnie,
                            candy=candy,
                            MNIST=mnist)
DEBUG = False
if not DEBUG:
    app.app.run(host='0.0.0.0', ssl_context='adhoc')  # so it can be assessed through network
else:
    fname = "/home/nvidia/Desktop/2.jpg"
    DEBUG = 'pose'
    if DEBUG == 'yolo':
        im = imread(fname)
    elif DEBUG == 'pose':
        im = imread(fname)
    elif DEBUG == 'cgan':
        im = Image.open(fname).convert('RGB')
        # img = img_preprocess(im, dtype='PIL', mode='RGB')

    outim = mods[DEBUG].process(im)
    image = cv2.cvtColor(np.asarray(outim), cv2.COLOR_RGB2BGR)
    cv2.imwrite('/home/nvidia/Desktop/result.jpg', image)
    print(outim.shape)
