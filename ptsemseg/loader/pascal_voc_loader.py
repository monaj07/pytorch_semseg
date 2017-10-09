import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
from PIL import Image

from tqdm import tqdm
from torch.utils import data

def get_data_path(name):
    js = open('config.json').read()
    data = json.loads(js)
    return data[name]['data_path']

class pascalVOCLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=224, image_transform=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 21
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.image_transform = image_transform

	file_list = []

	with open(root + '/ImageSets/Segmentation/' + split + '.txt', 'r') as f:
	    lines = f.readlines()
	filenames = [l.strip() for l in lines]
	N = len(filenames)
	print('Loading image and label filenames...\n')
	for i in range(N):
	    file_list.append(filenames[i])
	self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = self.root + '/JPEGImages/' + img_name + '.jpg'
        lbl_path = self.root + '/SegmentationClass/pre_encoded/' + img_name + '.png'

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl


    def transform(self, img, lbl):
        if self.image_transform is None:
            img = img[:, :, ::-1]
            img = img.astype(np.float64)
            img -= self.mean
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
            img = img.astype(float) / 255.0
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
        else:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC).astype(np.int32)
            img = self.image_transform(img)
            # img = img.transpose(2, 0, 1)  ## No need to do this transpose here anymore,
            # as it is done internally within ToTensor() function, inside image_transform.
            # img = torch.from_numpy(img).float() # It has already been converted to tensor using transforms.ToTensor() in image_transform.

        lbl[lbl==255] = -1
        if self.split=='train':
            lbl = lbl.astype(float)
            lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
            lbl = lbl.astype(int)
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def get_pascal_labels(self):
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                              [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]])


    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask


    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


if __name__ == '__main__':
    local_path = '/home/gpu_users/meetshah/segdata/pascal/VOCdevkit/VOC2012'
    dst = pascalVOCLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.imshow(dst.decode_segmap(labels.numpy()[i+1]))
            plt.show()
