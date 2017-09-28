import torch
import torch.nn as nn
import torch.nn.init as init
import os

ngf = 64
nz = 100

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 7 x 7
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 14 x 14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 28 x 28
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 56 x 56
            nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 112 x 112
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 224 x 224
        )

    def forward(self, input):
        output = self.main(input)
        return output

    def initialize(self, init_file=None):
        if (init_file):
            self.load_state_dict(init_file)
        else:
            self.apply(weights_init)


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.classifier= nn.Sequential(
            nn.Conv2d(4096, 1, 1), # Is this the right way, or we'd better use a linear layer???
            # The data shape is now (,1)
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.classifier(x)
        return y.view(-1, 1).squeeze(1)

    def initialize(self, init_file=None):
        if (init_file):
            self.load_state_dict(init_file)
        else:
            self.apply(weights_init)