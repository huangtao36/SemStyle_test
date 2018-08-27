# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is ...
"""
import os
import torch
from PIL import Image
import torchvision.transforms as transforms


def get_image_reader(dirpath, batch_size, workers=4):
    transform = set_transform()

    image_reader = torch.utils.data.DataLoader(
        ImageTestFolder(dirpath, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return image_reader


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

def has_image_ext(path):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    ext = os.path.splitext(path)[1]
    if ext.lower() in IMG_EXTENSIONS:
        return True
    return False


def list_image_folder(root):
    images = []
    dir_ = os.path.expanduser(root)
    for target in sorted(os.listdir(dir_)):
        d = os.path.join(dir_, target)
        if os.path.isdir(d):
            continue
        if has_image_ext(d):
            images.append(d)
    return images


def safe_pil_loader(path, from_memory=False):
    try:
        if from_memory:
            img = Image.open(path)
            res = img.convert('RGB')
        else:
            with open(path, 'rb') as f:
                img = Image.open(f)
                res = img.convert('RGB')
    except:
        res = Image.new('RGB', (299, 299), color=0)

    return res


class ImageTestFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.loader = safe_pil_loader
        self.transform = transform

        self.samples = list_image_folder(root)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, path

    def __len__(self):
        return len(self.samples)
