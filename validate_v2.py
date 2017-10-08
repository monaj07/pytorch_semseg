import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores

import pdb

############################################
# Padding layer to be applied to the input data
class padder_layer(nn.Module):
    def __init__(self, pad_size):
        super(padder_layer, self).__init__()
        self.apply_padding = nn.ConstantPad2d(pad_size, 0)  # Pad with zero values

    def forward(self, x):
        output = self.apply_padding(x)
        return output
############################################

def validate(args):
    ###################
    supervised = args.supervised # If set in the command line, it is a boolean with a value of True.
    ###################

    ############################################
    if supervised:
        # When pre_trained = 'gt', i.e. when using supervised image-net weights, images were normalized with image-net mean.
        image_transform = None
    else:
        # When pre_trained = 'self' or 'no', i.e. in the self-supervised case, or unsupervised case, the input images are normalized this way:
        image_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True,
                         img_size=(args.img_rows, args.img_cols), image_transform=image_transform)
    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4)

    ############################################
    # Load the trained Model
    assert(args.netF != '' and args.netS != '')
    print('\n' + '-' * 60)
    netF = torch.load(args.netF)
    netS = torch.load(args.netS)
    print('Loading the trained networks and for evaluation.')
    print('-' * 60)

    ############################################
    padder = padder_layer(pad_size=100)
    netF.eval() # Eval() function should be applied to the model at the evaluation phase
    netS.eval()
    padder.eval()

    ############################################
    # Porting the networks to CUDA
    if torch.cuda.is_available():
        netF.cuda(args.gpu)
        netS.cuda(args.gpu)
        padder.cuda(args.gpu)

    ############################################
    # Evaluation:
    gts, preds = [], []
    for i, (images, labels) in tqdm(enumerate(valloader)):
        ######################
        # Porting the data to Autograd variables and CUDA (if available)
        if torch.cuda.is_available():
            images = Variable(images.cuda(args.gpu))
            labels = Variable(labels.cuda(args.gpu))
        else:
            images = Variable(images)
            labels = Variable(labels)

        ######################
        # Passing the data through the networks and Computing the score maps
        padded_images = padder(images)
        feature_maps = netF(padded_images)
        score_maps = netS(feature_maps)
        outputs = F.upsample(score_maps, labels.size()[1:], mode='bilinear')
        pred = outputs.data.max(1)[1].cpu().numpy()

        gt = labels.data.cpu().numpy()
        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    ######################
    # Computing the mean-IoU and other scores
    score, class_iou = scores(gts, preds, n_class=n_classes)

    for k, v in score.items():
        print k, v

    for i in range(n_classes):
        print i, class_iou[i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--netF', nargs='?', type=str, required=True,
                        help='path to the netF model')
    parser.add_argument('--netS', nargs='?', type=str, required=True,
                        help='path to the netS model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=320,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=320,
                        help='Height of the input image')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--supervised', action='store_true',
                        help='Uses Imagenet normalization that is used in supervised pre-training')
    parser.add_argument('--split', nargs='?', type=str, default='val',
                        help='Split of dataset to test on')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to use')
    parser.add_argument('--save_folder', nargs='?', type=str, default='saved',
                        help='Where to retrieve the models')
    args = parser.parse_args()
    validate(args)

