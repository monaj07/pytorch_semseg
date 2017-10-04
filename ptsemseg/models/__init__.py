import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.linknet import *



def get_model(name, n_classes, pre_trained=True):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)
        if pre_trained:
            vgg16 = models.vgg16(pretrained=True)
            model.init_vgg16_params(vgg16)

    elif name == 'alexfcn':
        model = model(n_classes=n_classes)
        if pre_trained:
            alexnet = models.alexnet(pretrained=True)
            model.init_alex_params(alexnet)

    elif name == 'alexfcnv2':
        model_features = model()
        model_segmenter = alexnet_segmenter(n_classes=n_classes)
        if pre_trained:
            alexnet = models.alexnet(pretrained=True)
            model_features.init_alex_params(alexnet)
            model_segmenter.init_alex_params(alexnet)
        model = (model_features, model_segmenter)

    elif name == 'segnet':
        model = model(n_classes=n_classes, is_unpooling=True)
        if pre_trained:
            vgg16 = models.vgg16(pretrained=True)
            model.init_vgg16_params(vgg16)

    elif name == 'unet':
        model = model(n_classes=n_classes, is_batchnorm=True, in_channels=3, is_deconv=True)
    else:
        raise 'Model {} not available'.format(name)

    return model

def _get_model_instance(name):
    return {
        'alexfcnv2': alexnet_features,
        'alexfcn': alexfcn,
        'fcn32s': fcn32s,
        'fcn8s': fcn8s,
        'fcn16s': fcn16s,
        'unet': unet,
        'segnet': segnet,
        'pspnet': pspnet,
        'linknet': linknet,
    }[name]
