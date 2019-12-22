
from basic.dataset import camelyon_data
from ..dataset.dynamic_dataset import *
import torchvision.models as models
import torchvision.transforms as transforms
# 血液细胞评测
from basic.eval_validate import basic_validate
import torch.nn.functional as F
from basic import model
import random
import torch
from torch import nn
from torch import utils
from torch.autograd import Variable
from basic.utils import accuracy
from basic.utils import counter
from basic.utils import logs
from basic.utils import image_transform
from basic.utils import eval_method
import time
import pdb
import os
import numpy as np


class Validate(basic_validate.BasicValidate):
    """包含每一轮validate，BreastPathQ数据集的validation数据集只有185张图，返回p_k即可
    """

    def __init__(self, config, workspace):
        """这里只解析和训练相关的参数，不对模型进行加载"""
        super(Validate, self).__init__()
        self.config = config
        self.workspace=workspace
        self.log = logs.Log(os.path.join(self.workspace, "validate.txt"))


    def cfg(self, name):
        """获取配置简易方式"""
        return self.config.get_config('validate', name)

    def load_data(self):
        validate_transform = transforms.Compose([image_transform.RandomScale(shorter_side_range=(224, 224)),
                                                 transforms.RandomCrop(size=(224, 224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.783, 0.636, 0.74), (0.168, 0.187, 0.144))])


        validate_dataset = ValidDataset(self.config.get_config('validate', 'tumor_list'),
                                   self.config.get_config('validate', 'normal_list'),
                                   transform=validate_transform,
                                   tif_folder=self.config.get_config('base', 'train_tif_folder'),
                                   patch_size=self.config.get_config('base', 'patch_size'))
        # test dataset
        # iter_data = iter(validate_dataset)
        # _input, _labels, path_list = next(iter_data)
        return torch.utils.data.DataLoader(validate_dataset, batch_size=self.cfg('batch_size'),
                                           shuffle=False, num_workers=self.cfg('num_workers'))

    def run(self, _model, epoch):
        """单个epoch的validate
        :return 本次epoch中的p_k，其他不返回
        """
        #         pdb.set_trace()
        _model.eval()
        criterion = nn.CrossEntropyLoss()
        time_counter = counter.Counter()
        time_counter.addval(time.time(), key='validate epoch start')
        truth_labels = []
        pred_labels = []
        acc = {'avg_counter_total': counter.Counter(), 'avg_counter_pos': counter.Counter(),
               'avg_counter_neg': counter.Counter()}
        losses = counter.Counter()
        for i, data in enumerate(self.load_data(), 0):
            _input, _labels, path_list = data
            # forward and step
            if torch.cuda.is_available():
                _input = Variable(_input.type(torch.cuda.FloatTensor))
            else:
                _input = Variable(_input.type(torch.FloatTensor))
            _output = _model(_input)
            _output = F.softmax(_output)[:, 1].detach()
            _output = _output.cpu()
            loss = criterion(_output, _labels)
            acc_batch_total, acc_batch_pos, acc_batch_neg = accuracy.acc_binary_class(_output, _labels, 0.5)
            acc_batch = acc_batch_total
            acc['avg_counter_total'].addval(acc_batch_total)
            acc['avg_counter_pos'].addval(acc_batch_pos)
            acc['avg_counter_neg'].addval(acc_batch_neg)
            losses.addval(loss.item(), len(_output))
            time_counter.addval(time.time())

        return acc['avg_counter_total'].avg,acc['avg_counter_pos'].avg,acc['avg_counter_neg'].avg,losses.avg
