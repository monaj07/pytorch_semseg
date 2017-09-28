import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.models.gan_models import _netD, _netG
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.models.utils import padder_layer
from ptsemseg.metrics import scores
from lr_scheduling import *

nz = 100

def train(args):
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)

    # Setup Model
    net_features, net_segmenter = get_model(name=args.arch, n_classes=n_classes)
    netD = _netD()
    netG = _netG(nz=nz)
    padder = padder_layer(pad_size=100)

    criterion_gan = nn.BCELoss()

    input = torch.FloatTensor(args.batch_size_gan, 3, args.img_rows, args.img_cols)
    noise = torch.FloatTensor(args.batch_size_gan, nz, 1, 1)
    fixed_noise = torch.FloatTensor(args.batch_size_gan, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(args.batch_size_gan)

    real_label = .9
    fake_label = .1

    G = args.gpu

    if torch.cuda.is_available():
        input, label = input.cuda(G), label.cuda(G)
        noise, fixed_noise = noise.cuda(G), fixed_noise.cuda(G)

    fixed_noise = Variable(fixed_noise)

    if torch.cuda.is_available():
        net_features.cuda(G)
        net_segmenter.cuda(G)
        padder.cuda(G)
        netD.cuda(G)
        netG.cuda(G)
        criterion_gan.cuda(G)


    optimizerS = torch.optim.SGD(list(net_features.parameters()) + list(net_segmenter.parameters()), lr=args.l_rate,
                                momentum=0.99, weight_decay=5e-4)
    optimizerD = torch.optim.Adam(list(net_features.parameters()) + list(netD.parameters()), lr=args.l_rate,
                            betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.l_rate, betas=(args.beta1, 0.999))

    SS = 1
    GAN = 0

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(G))
                labels = Variable(labels.cuda(G))
            else:
                images = Variable(images)
                labels = Variable(labels)

            # iter = len(trainloader)*epoch + i
            # poly_lr_scheduler(optimizer, args.l_rate, iter)

            optimizerS.zero_grad()
            padded_images = padder(images)
            features = net_features(padded_images)
            outputs = net_segmenter(features)

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizerS.step()

            if (i + 1) % 20 == 0:
                print("Iter [%d/%d], Epoch [%d/%d] Loss: %.4f" % (
                i + 1, len(trainloader), epoch + 1, args.n_epoch, loss.data[0]))

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))

        torch.save((net_features, net_segmenter), "{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--batch_size_gan', nargs='?', type=int, default=1,
                        help='Batch Size for GAN')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--gpu', nargs='?', type=int, default=0,
                        help='which gpu to use')
    parser.add_argument('--lr_gan', type=float, default=0.0002, help='GAN learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    args = parser.parse_args()
    train(args)
