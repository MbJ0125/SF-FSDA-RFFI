import argparse
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
# dataset.py 顶部
try:
    import cv2  # 只给图像数据集用
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

import torchvision
from torch.utils.data import DataLoader

def make_dataset(image_list, dset_root):
    assert isinstance(image_list, list) and len(image_list) != 0
    images = []
    for line in image_list:
        item = line.strip().split()
        img_path = os.path.join(dset_root, item[0])
        img_label = int(item[1])
        images.append([img_path, img_label])
        assert os.path.exists(img_path), "[WRONG] %s does not exist" % img_path
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, imgs, transform=None,  mode='RGB'):
        
        self.imgs = imgs
        self.transform = transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index

    def __len__(self):
        return len(self.imgs)

def get_dataset(dset_file, dset_root): 
    ## prepare data
    txt_dset = open(dset_file).readlines()
    return make_dataset(txt_dset, dset_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data set')
    
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    args = parser.parse_args()

    dset_file_path = "./data/train_list.txt"
    mode="train" #validation
    args.dset_root = os.path.join("/home/chenhui/Code/dataset/VISDA-C", mode)
    dataset = get_dataset(dset_file_path, args.dset_root)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=False)

    print("load %d images from %s"%(len(dataset), args.dset_root))

    dset_file_path = "./data/validation_list.txt"
    mode="validation"
    args.dset_root = os.path.join("/home/chenhui/Code/dataset/VISDA-C", mode)
    dataset = get_dataset(dset_file_path, args.dset_root)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=False)

    print("load %d images from %s"%(len(dataset), args.dset_root))

def _ensure_ncl(x: np.ndarray) -> np.ndarray:
    """把 [N, L, C] 转为 [N, C, L]；若已是 [N, C, L] 则原样返回。"""
    x = np.squeeze(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    # 判定哪个轴是通道
    if x.shape[1] in (1, 2) and x.shape[2] > 8:   # [N,C,L]
        return x
    if x.shape[2] in (1, 2) and x.shape[1] > 8:   # [N,L,C]
        return np.transpose(x, (0, 2, 1))
    # 兜底当作 [N,L,C]
    return np.transpose(x, (0, 2, 1))

class IQNPYDataset(Dataset):
    """
    直接读取成批 .npy 数组：
      x_path: x_train_2ft.npy / x_test_2ft.npy, 形状 [N, C, L] 或 [N, L, C]
      y_path: y_train_2ft.npy / y_test_2ft.npy, 形状 [N]
    transform: 可传入对 (tensor) 的变换，例如归一化、加噪等；默认不变换
    """
    def __init__(self, x_path: str, y_path: str, transform=None):
        assert os.path.exists(x_path), f"{x_path} not found"
        assert os.path.exists(y_path), f"{y_path} not found"
        x = np.load(x_path)
        y = np.load(y_path)
        x = _ensure_ncl(x)                         # -> [N, C, L]
        y = np.squeeze(y).astype(np.int64)         # -> [N]
        assert x.shape[0] == y.shape[0], f"N mismatch: {x.shape[0]} vs {y.shape[0]}"
        self.x = torch.from_numpy(x).float()       # [N, C, L]
        self.y = torch.from_numpy(y).long()        # [N]
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        xi = self.x[idx]           # [C, L]  tensor
        yi = self.y[idx].item()    # int
        if self.transform:
            xi = self.transform(xi)
        return xi, yi, idx

# 可选：最简单的“恒等变换”，保持接口统一
def iq_train_transform():
    return lambda t: t             # 需要时你可以在这里做归一化/增广

def iq_test_transform():
    return lambda t: t