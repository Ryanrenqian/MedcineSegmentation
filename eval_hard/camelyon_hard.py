import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from  torch.nn import functional as F
from torch.autograd import Variable
import time
import torch.nn.functional as F

from ..eval_hard import BasicHard
from ..utils import image_transform
from ..dataset import camelyon_data
from ..utils import logs
from ..utils import counter
from ..utils import accuracy
from ..model import camelyon_models
from ..dataset import DynamicDataset, EvalDataset, ListDataset
import pdb
import json


class Hard(BasicHard):
    """Hard Minning, 提取难样本用于finetune
    内部不加载模型，只通过load model处理"""

    def __init__(self, config):
        super(Hard, self).__init__()
        self.config = config
        save_folder = os.path.join(self.config.get_config('base', 'save_folder'),
                                   self.config.get_config('base', 'last_run_date'))
        self.log = logs.Log(os.path.join(save_folder, "log.txt"))
        # test params
        self.best_epoch = 0
        self.best_acc = 0
        self.after_model_output = getattr(camelyon_models, 'after_model_output')

    def cfg(self, name):
        """获取配置简易方式"""
        return self.config.get_config('test', name)

    def checkpoint(self, model, save_helper):
        save_folder = os.path.join(save_helper.save_folder, 'models')
        epoch = self.config.get_config('test', 'epoch')
        checkpoint = os.path.join(save_folder, f'epoch_{epoch}_type_train_model.pth')
        epoch_checkpoint = torch.load(checkpoint)
        model.load_state_dict(epoch_checkpoint['model_state'])
        return model

    def load_normal_data(self):
        _size = self.config.get_config('base', 'crop_size')
        transform = image_transform.get_test_transforms(shorter_side_range=(_size, _size), size=(_size, _size))
        dataset = ListDataset(self.config.get_config('train', 'normal_list'),
                                transform=transform,
                                all_class=0,
                                tif_folder=self.config.get_config('base', self.config.get_config('base', 'train_tif_folder')),
                                patch_size=self.config.get_config('base', 'patch_size'))
        # 这里为了减少openslide的内存消耗，采用shuffle=False的方式，按顺序取数据
        return torch.utils.data.DataLoader(dataset, batch_size=self.cfg('batch_size'),
                                           shuffle=False, num_workers=self.cfg('num_workers'))

    def load_hard_data(self,filename):
        _size = self.config.get_config('base', 'crop_size')
        test_transform = image_transform.get_test_transforms(shorter_side_range=(_size, _size), size=(_size, _size))
        dataset = ListDataset(filename,
                                transform=test_transform,
                                all_class=0,
                                tif_folder=self.config.get_config('base', self.config.get_config('base','train_tif_folder')),
                                patch_size=self.config.get_config('base', 'patch_size'))
        # 这里为了减少openslide的内存消耗，采用shuffle=False的方式，按顺序取数据
        return torch.utils.data.DataLoader(dataset, batch_size=self.cfg('batch_size'),
                                           shuffle=False, num_workers=self.cfg('num_workers'))

    def hard(self, model,save_helper):
        '''
        动态的加载模型用于训练
        Parameters
        ----------
        _model
        epoch
        save_helper

        Returns
        -------

        '''
        model = self.checkpoint(model, save_helper)
        model.eval()
        time_counter = counter.Counter()
        time_counter.addval(time.time(), key='test epoch start')
        acc = {'avg_counter_total': counter.Counter(), 'avg_counter_pos': counter.Counter(),
               'avg_counter_neg': counter.Counter(),
               'avg_counter': counter.Counter(), 'epoch_acc_image': []}
        records = []
        for i, data in enumerate(self.load_normal_data(), 0):
            input_imgs, class_ids, patch_names=data
            output = model(input_imgs)
            output = output.cpu()
            class_ids = Variable(class_ids.type(torch.FloatTensor))
            output = F.softmax(output)[:, 1]
            acc_batch_total, acc_batch_pos, acc_batch_neg = accuracy.acc_binary_class(output.cpu(),class_ids, 0.5)
            acc_batch = acc_batch_total
            acc['avg_counter_total'].addval(acc_batch_total)
            acc['avg_counter_pos'].addval(acc_batch_pos)
            acc['avg_counter_neg'].addval(acc_batch_neg)
            acc['avg_counter'].addval(acc_batch)
            time_counter.addval(time.time())
            self.log.info(f"acc-iter/avg:[{acc_batch:.2f}/{acc['avg_counter'].avg:.2f}]-[{acc['avg_counter_total'].avg:.2f}--{acc['avg_counter_pos'].avg:.2f}--{acc['avg_counter_neg'].avg:.2f}], time consume:{time_counter.interval():.2f} s\r" )
            for i,patch_name in zip(output,patch_names):
                if i>0.5:
                    records.append(patch_names+'\n')
        # Save HardExamples file
        hardexamples=save_helper.save_hard_example('hardexample.list',records)
        time_counter.addval(time.time(), key='End seeking hard exmample')
        # Fine tune models
        model.train()
        criterion = nn.CrossEntropyLoss()
        losses = counter.Counter()
        for epoch in range(self.config.get_config('hard','finetune','epoch'))
            for i,data in enumerate(self.load_hard_data(), 0):
                input, labels, path_list = data
                if torch.cuda.is_available():
                    input = Variable(input.type(torch.cuda.FloatTensor))
                else:
                    input = Variable(input.type(torch.FloatTensor))
                output = model(input).squeeze().cpu()
                #             pdb.set_trace()
                loss = criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #             pdb.set_trace()
                output = F.softmax(output)[:, 1].detach()
                acc_batch_total, acc_batch_pos, acc_batch_neg = accuracy.acc_binary_class(output, labels,0.5)
            save_helper.save_epoch_model( epoch, "hard", acc, losses, model, None)




