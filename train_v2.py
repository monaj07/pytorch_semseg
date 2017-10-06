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
from torchvision import transforms

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
# Padding layer to be applied to the input data
class padder_layer(nn.Module):
    def __init__(self, pad_size):
        super(padder_layer, self).__init__()
        self.apply_padding = nn.ConstantPad2d(pad_size, 0)  # Pad with zero values

    def forward(self, x):
        output = self.apply_padding(x)
        return output
############################################

############################################
# Parameter initialization for the network layers
# It should be applied to the network using `apply` function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
############################################


def train(args):
    ###################
    pre_trained = args.pre_trained # default='gt'
    ###################

    ############################################
    if pre_trained == 'gt':
        # When pre_trained = 'gt', i.e. when using supervised image-net weights, images were normalized with image-net mean.
        image_transform = None
    else:
        # When pre_trained = 'self' or 'no', i.e. in the self-supervised case, or unsupervised case, the input images are normalized this way:
        image_transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True,
                         img_size=(args.img_rows, args.img_cols), image_transform=image_transform)
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)
    ############################################

    ############################################
    # Setup Model
    if args.netF_path == '' or args.netS_path == '':
        if pre_trained == 'gt':
            netF, netS = get_model(args.arch, n_classes, pre_trained=True)
            print('*' * 60)
            print('*' * 60)
            print('*' * 60)
            print('Training using a GT-supervised pre-trained model ...')
            print('*' * 60)
            print('*' * 60)
            print('*' * 60 + '\n')
        elif pre_trained == 'self':
            netF, netS = get_model(args.arch, n_classes, pre_trained=False)
            assert (args.model_path != '')
            # Loading a self-supervised trained model (ss_model)
            print('x' * 60)
            print('x' * 60)
            print('x' * 60)
            print('Training a using a self-supervised pre-trained model ...')
            ss_model = torch.load(args.model_path)
            netF.init_alex_params(ss_model)
            netS.init_alex_params(ss_model)
            print('Restored the self-supervised pre_trained model {}'.format(args.model_path))
            print('x' * 60)
            print('x' * 60)
            print('x' * 60 + '\n')
        else: # pre_trained == 'no':
            netF, netS = get_model(args.arch, n_classes, pre_trained=False)
            # Random weight initialization
            netF.apply(weights_init)
            netS.apply(weights_init)
            print('o' * 60)
            print('o' * 60)
            print('o' * 60)
            print('Training without any pre-trained model ...')
            print('o' * 60)
            print('o' * 60)
            print('o' * 60 + '\n')

    padder = padder_layer(pad_size=100)
    ############################################

    ############################################
    # If resuming the training from a saved model
    if args.netF_path != '':
        print('\n' + '-' * 40)
        netF = torch.load(args.netF_path)
        print('Resuming the training netF from model in {}'.format(args.netF_path))
    if args.netS_path != '':
        netS = torch.load(args.netS_path)
        print('Resuming the training netS from model in {}'.format(args.netS_path))
        print('-' * 40)
    ############################################

    ############################################
    # Porting the networks to CUDA
    if torch.cuda.is_available():
        netF.cuda(args.gpu)
        netS.cuda(args.gpu)
        padder.cuda(args.gpu)
    ############################################

    ############################################
    # Defining the optimizer over the network parameters
    optimizerSS = torch.optim.SGD([{'params': netF.features.parameters()},
                                        {'params': netF.classifier.parameters(), 'lr':10*args.l_rate},
                                        {'params': netS.parameters(), 'lr':20*args.l_rate}],
                                  lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    optimizerSS_init = copy.deepcopy(optimizerSS)
    ############################################

    ############################################
    # TRAINING:
    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
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
            netF.zero_grad()
            netS.zero_grad()

            ######################
            # Passing the data through the networks
            padded_images = padder(images)
            feature_maps = netF(padded_images)
            score_maps = netS(feature_maps)
            outputs = F.upsample(score_maps, labels.size()[1:], mode='bilinear')

            ######################
            # Computing the loss and doing back-propagation
            loss = cross_entropy2d(outputs, labels)
            loss.backward()

            ######################
            # Updating the parameters
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
    parser.add_argument('--pre_trained', nargs='?', type=str, default='gt',
                        help='network pre_trained? [\'gt, self, no\']')
    parser.add_argument('--model_path', nargs='?', type=str, default='',
                        help='path to the self-supervised trained model')
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
