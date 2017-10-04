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

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import scores
from lr_scheduling import *
import pdb


class padder_layer(nn.Module):
    def __init__(self, pad_size):
        super(padder_layer, self).__init__()
        self.apply_padding = nn.ConstantPad2d(pad_size, 0)  # Pad with zero values

    def forward(self, x):
        output = self.apply_padding(x)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    ############################################
    # Setup Model
    pre_trained = True
    netF, netS = get_model(args.arch, n_classes, pre_trained=pre_trained)
    padder = padder_layer(pad_size=100)
    ############################################
    # Random weight initialization
    if not pre_trained:
        netF.apply(weights_init)
        netS.apply(weights_init)
    ############################################

    if args.netF_path != '':
        print('\n' + '-' * 40)
        netF = torch.load("./{}/{}".format(args.save_folder, args.netF_path))
        print('Restored the trained network {}'.format(args.netF_path))
    if args.netS_path != '':
        netS = torch.load("./{}/{}".format(args.save_folder, args.netS_path))
        print('Restored the trained network {}'.format(args.netS_path))
        print('-' * 40)


    if torch.cuda.is_available():
        netF.cuda(args.gpu)
        netS.cuda(args.gpu)
        padder.cuda(args.gpu)

    optimizerSS = torch.optim.SGD(list(netF.parameters())+list(netS.parameters()), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(args.gpu))
                labels = Variable(labels.cuda(args.gpu))
            else:
                images = Variable(images)
                labels = Variable(labels)

            #iter = len(trainloader)*epoch + i
            adjust_learning_rate(optimizerSS, args.l_rate, epoch)

            optimizerSS.zero_grad()
            netF.zero_grad()
            netS.zero_grad()

            padded_images = padder(images)
            feature_maps = netF(padded_images)
            score_maps = netS(feature_maps)
            outputs = F.upsample(score_maps, labels.size()[1:], mode='bilinear')

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizerSS.step()


            if (i+1) % 20 == 0:
                print("Iter [%d/%d], Epoch [%d/%d] Loss: %.4f" % (i+1, len(trainloader), epoch+1, args.n_epoch, loss.data[0]))

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))
        if args.netF_path != '':
            torch.save(netF, "./{}/netF_{}_{}_{}_from_{}.pkl".format(args.save_folder, args.arch, args.dataset, epoch, args.netF_path))
        else:
            torch.save(netF, "./{}/netF_{}_{}_{}.pkl".format(args.save_folder, args.arch, args.dataset, epoch))
        if args.netS_path != '':
            torch.save(netS, "./{}/netS_{}_{}_{}_from_{}.pkl".format(args.save_folder, args.arch, args.dataset, epoch, args.netS_path))
        else:
            torch.save(netS, "./{}/netS_{}_{}_{}.pkl".format(args.save_folder, args.arch, args.dataset, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, required=True,
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=320,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=320,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--netF_path', nargs='?', type=str, default='',
                        help='path to the netF model')
    parser.add_argument('--netS_path', nargs='?', type=str, default='',
                        help='path to the netS model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to use')
    parser.add_argument('--split', nargs='?', type=str, default='train',
                        help='Split of dataset to test on')
    parser.add_argument('--save_folder', nargs='?', type=str, default='saved',
                        help='Where to save and retrieve the models')
    args = parser.parse_args()
    train(args)
