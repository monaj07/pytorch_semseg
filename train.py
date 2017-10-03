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
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import scores
from lr_scheduling import *
import pdb

def train(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, shuffle=True)


    # Setup Model
    model = get_model(args.arch, n_classes)

    if args.restore_from != '':
        print('\n' + '-' * 40)
        model = torch.load(args.restore_from)
        print('Restored the trained network {} '.format(args.restore_from))
        print('-' * 40)

    if torch.cuda.is_available():
        model.cuda(args.gpu)
        test_image, test_segmap = loader[0]
        test_image = Variable(test_image.unsqueeze(0).cuda(args.gpu))
    else:
        test_image, test_segmap = loader[0]
        test_image = Variable(test_image.unsqueeze(0))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    for epoch in range(args.n_epoch):
        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda(args.gpu))
                labels = Variable(labels.cuda(args.gpu))
            else:
                images = Variable(images)
                labels = Variable(labels)

            #iter = len(trainloader)*epoch + i
            adjust_learning_rate(optimizer, args.l_rate, epoch)
            
            optimizer.zero_grad()
            outputs = model(images)

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizer.step()


            if (i+1) % 20 == 0:
                print("Iter [%d/%d], Epoch [%d/%d] Loss: %.4f" % (i+1, len(trainloader), epoch+1, args.n_epoch, loss.data[0]))

        # test_output = model(test_image)
        # predicted = loader.decode_segmap(test_output[0].cpu().data.numpy().argmax(0))
        # target = loader.decode_segmap(test_segmap.numpy())

        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))
        if args.restore_from != '':
            torch.save(model, "{}_{}_from_{}_{}.pkl".format(args.arch, args.dataset, args.restore_from, epoch))
        else:
            torch.save(model, "{}_{}_{}.pkl".format(args.arch, args.dataset, epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
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
    parser.add_argument('--restore_from', nargs='?', type=str, default='',
                        help='path to the saved weights to use')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to use')
    args = parser.parse_args()
    train(args)
