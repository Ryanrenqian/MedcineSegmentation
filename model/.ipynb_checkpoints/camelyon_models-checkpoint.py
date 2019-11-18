"""包括blood cell image 中用到的所有模型"""
import torchvision.models as models
import torch.nn as nn
import torch
import sys
from basic.utils import Log
from basic.model import scannet
import pdb
# from  scannet import *

# model
def load_model(model, _model_name):
    """载入模型"""
    log = Log()
#     pdb.set_trace()
    _model_function = getattr(sys.modules[__name__], model)
    log.info("loading model:%s" % _model_name)
    return _model_function(_model_name, pretrained=False)


def after_model_output(output, config):
    # google net在统计时，使用aux
    if config.get_config('model', 'model_name') == "googlenet" and config.get_config('model', 'params', 'use_aux'):
        output = output.logits + 0.3 * output.aux_logits1 + 0.3 * output.aux_logits2
    if config.get_config('model', 'model_name') == "inception_v3" and config.get_config('model', 'params', 'use_aux'):
        output = output.logits + 0.3 * output.aux_logits

    return output


def alexnet(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    # _model.classifier[-1] = nn.Linear(in_features=_model.classifier[-1].in_features, out_features=4, bias=True)
    if torch.cuda.is_available():
        _model = torch.nn.DataParallel(_model).cuda()
    return _model


def resnet(model_name, pretrained=False):
#     pdb.set_trace()
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=1)
#     _model.fc = nn.Linear(in_features=_model.fc.in_features, out_features=4, bias=True)
    _model.fc = nn.Sequential(nn.Linear(in_features=_model.fc.in_features, out_features=1, bias=True), nn.Sigmoid())

    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def googlenet(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3, aux_logits=False)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def googlenet_aux(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    torch.nn.DataParallel(_model).cuda()
    return _model


def inception_no_aux(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3, aux_logits=False)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def inception_with_aux(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3, aux_logits=True)
    torch.nn.DataParallel(_model).cuda()
    return _model


def densenet121(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def densenet161(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def vgg11_bn(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def squeezenet1_0(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def squeezenet1_1(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def shufflenet(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def vgg19_bn(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model


def vgg19(model_name, pretrained=False):
    _model = models.__dict__[model_name](pretrained=pretrained, num_classes=3)
    _model = torch.nn.DataParallel(_model).cuda()
    return _model




def scannet_0(model_name, pretrained=False):
    _model = scannet.Scannet(3, 2)
    if torch.cuda.is_available():
        _model = torch.nn.DataParallel(_model).cuda()
    return _model
