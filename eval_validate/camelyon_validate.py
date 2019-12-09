from basic.data import camelyon_data
import torchvision.models as models
import torchvision.transforms as transforms
# 血液细胞评测
from basic.eval_validate import basic_validate
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

    def __init__(self, config, save_helper):
        """这里只解析和训练相关的参数，不对模型进行加载"""
        super(Validate, self).__init__()
        self.config = config
        self.save_helper = save_helper
        self.log = logs.Log(os.path.join(save_helper.save_folder, "log.txt"))
        # dataloader
        self.validate_loader = self.load_data()
        # validate params
        self.is_cuda = self.cfg('platform') == 'ubuntu'

    def cfg(self, name):
        """获取配置简易方式"""
        return self.config[name]

    def load_data(self):
        validate_transform = transforms.Compose([image_transform.RandomScale(shorter_side_range=(224, 224)),
                                                 transforms.RandomCrop(size=(224, 224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.783, 0.636, 0.74), (0.168, 0.187, 0.144))])
        validate_dataset = breast_images.BreastDataset(self.cfg('data_folder'), transform=validate_transform,
                                                       groundtruth_file='val_labels.csv')
        # test dataset
        # iter_data = iter(validate_dataset)
        # _input, _labels, path_list = next(iter_data)
        return torch.utils.data.DataLoader(validate_dataset, batch_size=self.cfg('batch_size'),
                                           shuffle=True, num_workers=self.cfg('num_workers'))

    def validate(self, _model, epoch):
        """单个epoch的validate
        :return 本次epoch中的p_k，其他不返回
        """
        #         pdb.set_trace()
        _model.eval()
        time_counter = counter.Counter()
        time_counter.addval(time.time(), key='validate epoch start')
        truth_labels = []
        pred_labels = []
        for i, data in enumerate(self.validate_loader, 0):
            _input, _labels, path_list = data
            # forward and step
            if self.is_cuda:
                _input = Variable(_input.type(torch.cuda.FloatTensor))
            else:
                _input = Variable(_input.type(torch.FloatTensor))
            _output = _model(_input)
            _output = _output.cpu()

            # 统计每一轮的信息
            acc_image_list = accuracy.topk_with_class(_output.cpu(), _labels, path_list, topk=1)
            time_counter.addval(time.time())

            """统计均值"""
            for index in range(len(acc_image_list)):
                truth_labels.append(float(_labels[index]))
                pred_labels.append(float(acc_image_list[index]['pred'][0]))

        time_counter.addval(time.time(), key='validate epoch end')
        p_k = eval_method.predprob(truth_labels, pred_labels)
        if max(pred_labels) == min(pred_labels):
            p_k = 0
            self.log.info('p_k error')
            self.log.info(pred_labels)
            self.log.info(p_k)

        self.log.info('\nvalidate epoch:%d, time consume:%.2f s, p_k:%.2f ' %
                      (epoch, time_counter.key_interval(key_ed='validate epoch end', key_st='validate epoch start'),
                       p_k))
        return p_k
