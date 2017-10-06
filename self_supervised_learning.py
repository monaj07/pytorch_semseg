"""
This is the code for Self-Supervised Learning.
The goal is to learn image feature maps using video data.
The idea is to assign the frames within each video a unique class label, and perform a classification.
"""

import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.init as init
import torchvision.datasets as dset
from torchvision import transforms

from networks import AlexNet

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import scores
from lr_scheduling import *
import pdb
import copy

############################################
# Parameter initialization for the network layers
# It should be applied to the network using `apply` function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
############################################


def train(args):
    ############################################
    # Setup Dataloader
    data_path = get_data_path(args.dataset)
    dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Scale(2*args.img_rows),
                                   transforms.RandomCrop(args.img_rows),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    n_classes = len(dataset.classes)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    ############################################

    ######## pretrained = FALSE ===>>> NO PRE-TRAINING ########
    ############################################
    # Setup Model
    """
    model = models.alexnet(pretrained=False)
    block = list(model.classifier.children())
    old_layer = block.pop()
    num_feats = old_layer.in_features
    block.append(nn.Linear(num_feats, n_classes))
    model.classifier = nn.Sequential(*block)
    """
    model = AlexNet(num_classes=n_classes)
    ############################################

    ############################################
    # Random weight initialization
    model.apply(weights_init)
    ############################################

    ############################################
    # If resuming the training from a saved model
    if args.model_path != '':
        print('\n' + '-' * 40)
        model = torch.load(args.model_path)
        print('Restored the trained network {}'.format(args.model_path))
    ############################################


    ############################################
    criterion = nn.CrossEntropyLoss() # Loss criterion
    # Porting the networks to CUDA
    if torch.cuda.is_available():
        model.cuda(args.gpu)
        criterion.cuda(args.gpu)
    ############################################

    ############################################
    # Defining the optimizer over the network parameters
    optimizerSS = torch.optim.SGD([{'params': model.features.parameters()},
                                        {'params': model.classifier.parameters(), 'lr':1*args.l_rate}],
                                  lr=args.l_rate, momentum=0.9, weight_decay=5e-4)
    optimizerSS_init = copy.deepcopy(optimizerSS)
    ############################################

    ############################################
    # TRAINING:
    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(dataloader):
            ######################
            # Porting the data to Autograd variables and CUDA (if available)
            if torch.cuda.is_available():
                images = Variable(images.cuda(args.gpu))
                labels = Variable(labels.cuda(args.gpu))
            else:
                images = Variable(images)
                labels = Variable(labels)

            ######################
            # Scheduling the learning rate
            #adjust_learning_rate(optimizerSS, args.l_rate, epoch)
            adjust_learning_rate_v2(optimizerSS, optimizerSS_init, epoch, step=20)

            ######################
            # Setting the gradients to zero at each iteration
            optimizerSS.zero_grad()
            model.zero_grad()

            ######################
            # Passing the data through the network
            outputs = model(images)

            ######################
            # Computing the loss and doing back-propagation
            loss = criterion(outputs, labels)
            loss.backward()

            ######################
            # Updating the parameters
            optimizerSS.step()

            if (i+1) % 20 == 0:
                print("Iter [%d/%d], Epoch [%d/%d] Loss: %.4f" % (i+1, len(dataloader), epoch+1, args.n_epoch, loss.data[0]))

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))
        if (epoch+1) % 5 == 0:
            if args.model_path != '':
                torch.save(model, "./{}/double_scale_model_{}_{}_{}_from_{}.pkl" .format(args.save_folder, args.arch, args.dataset, epoch, args.model_path))
            else:
                torch.save(model, "./{}/double_scale_model_{}_{}_{}.pkl".format(args.save_folder, args.arch, args.dataset, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, required=True,
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='videos',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=224,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=224,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=128,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-2,
                        help='Learning Rate')
    parser.add_argument('--model_path', nargs='?', type=str, default='',
                        help='path to the model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to use')
    parser.add_argument('--split', nargs='?', type=str, default='train',
                        help='Split of dataset to test on')
    parser.add_argument('--save_folder', nargs='?', type=str, default='saved_pretrained',
                        help='Where to save and retrieve the models')
    args = parser.parse_args()
    train(args)
