import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores

import pdb

class padder_layer(nn.Module):
    def __init__(self, pad_size):
        super(padder_layer, self).__init__()
        self.apply_padding = nn.ConstantPad2d(pad_size, 0)  # Pad with zero values

    def forward(self, x):
        output = self.apply_padding(x)
        return output


def validate(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4)

    ############################################
    # Load the trained Model
    assert(args.netF_path != '' and args.netS_path != '')
    print('\n' + '-' * 60)
    netF = torch.load(args.netF_path)
    netS = torch.load(args.netS_path)
    print('Loading the trained networks and for evaluation.')
    print('-' * 60)
    ############################################
    padder = padder_layer(pad_size=100)

    if torch.cuda.is_available():
        netF.cuda(args.gpu)
        netS.cuda(args.gpu)
        padder.cuda(args.gpu)

    gts, preds = [], []
    for i, (images, labels) in tqdm(enumerate(valloader)):
        if torch.cuda.is_available():
            images = Variable(images.cuda(args.gpu))
            labels = Variable(labels.cuda(args.gpu))
        else:
            images = Variable(images)
            labels = Variable(labels)

        padded_images = padder(images)
        feature_maps = netF(padded_images)
        score_maps = netS(feature_maps)
        outputs = F.upsample(score_maps, labels.size()[1:], mode='bilinear')

        pred = outputs.data.max(1)[1].cpu().numpy()
        pred = np.squeeze(pred)
        pred = cv2.resize(pred, labels.size()[1:][::-1], interpolation=cv2.INTER_NEAREST)
        pred = np.expand_dims(pred, axis=0)
        gt = labels.data.cpu().numpy()

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    score, class_iou = scores(gts, preds, n_class=n_classes)

    for k, v in score.items():
        print k, v

    for i in range(n_classes):
        print i, class_iou[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--netF_path', nargs='?', type=str, required=True,
                        help='path to the netF model')
    parser.add_argument('--netS_path', nargs='?', type=str, required=True,
                        help='path to the netS model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=320,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=320,
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='val',
                        help='Split of dataset to test on')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to use')
    parser.add_argument('--save_folder', nargs='?', type=str, default='saved',
                        help='Where to retrieve the models')
    args = parser.parse_args()
    validate(args)

