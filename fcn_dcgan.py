import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torch.autograd import Variable

from ptsemseg.models import get_model
from ptsemseg.models.gan_models import _netD, _netG
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.models.utils import padder_layer
from ptsemseg.metrics import scores
from lr_scheduling import *
import os

nz = 100

def train(args):
    # Setup Dataloader
    if args.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        assert(args.img_cols==args.img_rows)
        dataset = dset.ImageFolder(root=args.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Scale(args.img_cols),
                                       transforms.CenterCrop(args.img_cols),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif args.dataset in ['pascal', 'camvid']:
        data_loader = get_loader(args.dataset)
        data_path = get_data_path(args.dataset)
        dataset = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))

    n_classes = dataset.n_classes
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    ############################
    # Setup the Models
    net_features, net_segmenter = get_model(name=args.arch, n_classes=n_classes)
    netD = _netD()
    netG = _netG(nz=nz)
    padder = padder_layer(pad_size=100)
    ############################

    ############################
    ### Initialization:
    if args.netD != '':
        netD.load_state_dict(torch.load(os.path.join(args.outf, args.netD)))
    if args.netG != '':
        netG.load_state_dict(torch.load(os.path.join(args.outf, args.netG)))
    if args.net_features != '':
        netD.load_state_dict(torch.load(os.path.join(args.outf, args.net_features)))
    if args.net_segmenter != '':
        netD.load_state_dict(torch.load(os.path.join(args.outf, args.net_segmenter)))
    ############################

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
        for i, data in enumerate(trainloader):

            real_cpu, label_cpu = data
            batch_size = real_cpu.size(0)
            if torch.cuda.is_available():
                real_cpu = real_cpu.cuda(G)
                label_cpu = label_cpu.cuda(G)
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)
            labelv_semantic = Variable(label_cpu)

            # iter = len(trainloader)*epoch + i
            # poly_lr_scheduler(optimizer, args.l_rate, iter)

            if GAN:
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                optimizerD.zero_grad()
                net_features.zero_grad()
                netD.zero_grad()

                # train with real
                label.resize_(batch_size).fill_(real_label)
                labelv = Variable(label)
                inputv_feats = net_features(inputv)
                output = netD(inputv_feats)
                errD_real = criterion_gan(output, labelv)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev)
                labelv = Variable(label.fill_(fake_label))
                fake_feats = net_features(fake.detach())
                output = netD(fake_feats)
                errD_fake = criterion_gan(output, labelv)
                errD_fake.backward()
                D_G_z1 = output.data.mean()

                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                optimizerG.zero_grad()
                netG.zero_grad()

                label.resize_(batch_size).fill_(real_label)
                labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                noise2v = Variable(noise)
                fake2 = netG(noise2v)
                fake_feats2 = net_features(fake2)
                output = netD(fake_feats2)
                errG = criterion_gan(output, labelv)
                errG.backward()
                D_G_z2 = output.data.mean()
                optimizerG.step()

            if SS:
                optimizerS.zero_grad()
                net_features.zero_grad()
                net_segmenter.zero_grad()

                padded_images = padder(inputv)
                features = net_features(padded_images)
                outputs = net_segmenter(features)

                loss = cross_entropy2d(outputs, labelv_semantic)

                loss.backward()
                optimizerS.step()

            ######################################################
            ### Loss Report:
            ######################################################

            if SS:
                if (i + 1) % 20 == 0:
                    print("Iter [%d/%d], Epoch [%d/%d] Loss: %.4f" % (
                        i + 1, len(trainloader), epoch + 1, args.n_epoch, loss.data[0]))

            if GAN:
                if (i) % 100 == 0 and i > 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (epoch, args.n_epoch, i, len(trainloader),
                             errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                if i % 500 == 0 and i > 0:
                    vutils.save_image(real_cpu,
                                      '%s/real_samples.png' % args.outf,
                                      normalize=True)
                    fake = netG(fixed_noise)
                    vutils.save_image(fake.data,
                                      '%s/fake_samples_epoch_%03d_%03d.png' % (args.outf, epoch, i),
                                      normalize=True)
                if i % 3000 == 0 and i > 0:
                    torch.save(net_features.state_dict(), '%s/net_features_epoch_%d_%d.pth' % (args.outf, epoch, i))
                    torch.save(netG.state_dict(), '%s/netG_epoch_%d_%d.pth' % (args.outf, epoch, i))
                    torch.save(netD.state_dict(), '%s/netD_epoch_%d_%d.pth' % (args.outf, epoch, i))

        ######################################################
        ### Do checkpointing:
        ######################################################
        torch.save((net_features, net_segmenter),
                   "{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))
        continue
        torch.save(net_features.state_dict(), '%s/net_features_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
        #torch.save((net_features, net_segmenter), "{}_{}_{}_{}.pkl".format(args.arch, args.dataset, args.feature_scale, epoch))


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
    parser.add_argument('--lr_gan', type=float, default=0.0002,
                        help='GAN learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--netG', default='',
                        help="path to netG (to continue training)")
    parser.add_argument('--netD', default='',
                        help="path to netD (to continue training)")
    parser.add_argument('--outf', default='./saved',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int,
                        help='manual seed')
    parser.add_argument('--net_features', default='',
                        help="path to feature_net (to continue training)")
    parser.add_argument('--net_segmenter', default='',
                        help="path to segmenter_net (to continue training)")

    args = parser.parse_args()
    train(args)
