from basic.data import camelyon_data,dynamic_dataset
import torchvision.models as models
import torchvision.transforms as transforms
# 血液细胞评测
from basic.eval_train import basic_train
from basic.model import camelyon_models
from basic import model
import random
import torch
from torch import nn
from torch import utils
from torch.autograd import Variable
import torch.optim as optim
from basic.utils import accuracy
from basic.utils import counter
from basic.utils import logs
from basic.utils import image_transform
import time
import pdb
import os
import json
import torch.nn.functional as F


class Train(basic_train.BasicTrain):
    """包含每一轮train，返回每次batch_size的output
    """

    def __init__(self, config):
        """这里只解析和训练相关的参数，不对模型进行加载"""
        super(Train, self).__init__()
        self.config = config
        save_folder = os.path.join(self.config.get_config('base', 'save_folder'),
                                   self.config.get_config('base', 'last_run_date'))
        self.log = logs.Log(os.path.join(save_folder, "log.txt"))
        self.train_loader = self.load_data()
        # self.valid_loader = self.load_validation()
        self.after_model_output = getattr(camelyon_models, 'after_model_output')

    def cfg(self, name):
        """获取配置简易方式"""
        return self.config.get_config('train', name)

    def reload_data(self):
        """重新导入数据"""
        #         pdb.set_trace()
        self.train_loader = self.load_data()

    def get_train_data(self):
        _size = self.config.get_config('base', 'crop_size')
        train_transform = image_transform.get_train_transforms(shorter_side_range = (_size, _size), size = (_size, _size))
        if self.config.get_config('train','method') == 'base':
            train_dataset = camelyon_data.EvalDataset(self.config.get_config('train', 'train_list'),
                                                      transform=train_transform,
                                                      tif_folder=self.config.get_config('base', 'train_tif_folder'),
                                                      patch_size=self.config.get_config('base', 'patch_size'))
        elif self.config.get_config('train','method') == 'on_the_fly':
            dataset = dynamic_dataset.DynamicDataset(self.config.get_config('train', 'tumor_list'),
                                                     self.config.get_config('train', 'normal_list'),
                                                     transform=train_transform,
                                                     data_size=self.config.get_config('train','data_size'),
                                                     replacement=self.config.get_config('train','replacement'),
                                                     tif_folder=self.config.get_config('base', 'train_tif_folder'),
                                                     patch_size=self.config.get_config('base', 'patch_size'))
            train_dataset =dataset.sample()
        self.log.info("update dataset")
        return train_dataset

    def load_data(self):
        train_dataset =  self.get_train_data()
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg('batch_size'),
                                    shuffle=True, num_workers=self.cfg('num_workers'))

    def init_optimizer(self, _model):
        _params = self.cfg('params')
        self.optimizer = torch.optim.SGD(_model.parameters(), lr=_params['lr_start'], momentum=_params['momentum'],
                                         weight_decay=_params['weight_decay'])
        # self.optimizer_schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=_params['lr_decay_epoch'],
        #                                                     gamma=_params['lr_decay_factor'], last_epoch=10)

    def train(self, _model,  hard_mining_times, save_helper,config,writer,validation):
        """单个epoch的train 参数在epoch中是原子操作
        :过程保存：1.将单轮epoch中对每个样本的分类情况记录下来 2.将模型通过checkpoint保存
        :
        :return 本次epoch中的所有样本的详细结果，平均acc，loss
        """
        _model.train()

        criterion = nn.CrossEntropyLoss()

        #         pdb.set_trace()
        acc = {'avg_counter_total': counter.Counter(), 'avg_counter_pos': counter.Counter(),
               'avg_counter_neg': counter.Counter(),
               'avg_counter': counter.Counter(), 'epoch_acc_image': []}
        train_epoch_start = 0
        train_epoch_stop = config.get_config("train" ,'total_epoch')
        if config.get_config("train","resume","run_this_module"):
            train_epoch_start = config.get_config("train","resume","start_epoch")
            train_epoch_stop = train_epoch_start+config.get_config("train" ,"resume",'total_epoch')
        losses = counter.Counter()
        time_counter = counter.Counter()
        time_counter.addval(time.time(), key='training epoch start')
        iteration = 0
        for epoch in range(train_epoch_start, train_epoch_stop):
            for i, data in enumerate(self.load_data(), 0):
                iteration +=1
                train_input, train_labels, path_list = data
                if torch.cuda.is_available():
                    train_input = Variable(train_input.type(torch.cuda.FloatTensor))
                else:
                    train_input = Variable(train_input.type(torch.FloatTensor))
                train_output = _model(train_input).squeeze().cpu()
    #             pdb.set_trace()
                loss = criterion(train_output, train_labels) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    #             pdb.set_trace()
                train_output=F.softmax(train_output)[:,1].detach()
                acc_batch_total, acc_batch_pos, acc_batch_neg = accuracy.acc_binary_class(train_output, train_labels, 0.5)
                acc_batch = acc_batch_total
                writer.add_scalar('acc_batch_total in train',acc_batch,iteration)
                writer.add_scalar('acc_batch_total in train',acc_batch_total,iteration)
                writer.add_scalar('acc_batch_pos in train',acc_batch_pos,iteration)
                writer.add_scalar('acc_batch_neg in train',acc_batch_neg,iteration)
                writer.add_scalar('loss in train',loss.item(),iteration)
                writer.add_scalar('Lr',self.optimizer.state_dict()['param_groups'][0]['lr'])
                acc['avg_counter_total'].addval(acc_batch_total)
                acc['avg_counter_pos'].addval(acc_batch_pos)
                acc['avg_counter_neg'].addval(acc_batch_neg)
                acc['avg_counter'].addval(acc_batch)
                losses.addval(loss.item(), len(train_output))
                time_counter.addval(time.time())
                #             pdb.set_trace()
                if i % self.config.get_config('base', 'print_freq', 'batch_iter') == 0:
                    self.log.info(
                    'train new epoch:%d/%d,batch iter:%d/%d, lr:%.5f, acc-iter/avg:[%.2f/%.2f]-[avg:%.2f-pos:%.2f-neg:%.2f], loss:%.2f/%.2f,time consume:%.2f s' % (
                        epoch, self.config.get_config('train', 'total_epoch'), i, len(self.train_loader),
                        self.optimizer.state_dict()['param_groups'][0]['lr'], acc_batch,
                        acc['avg_counter'].avg,
                        acc['avg_counter_total'].avg, acc['avg_counter_pos'].avg, acc['avg_counter_neg'].avg,
                        loss.item(),
                        losses.avg,
                        time_counter.interval()), '\r')
#                     self.log.info(f'train new epoch:{epoch}/{self.config.get_config('train', 'total_epoch')},batch iter:{i}/{len(self.train_loader)}, lr:{self.optimizer.state_dict()['param_groups'][0]['lr']:.10f}, train_acc-iter/avg:[{acc_batch:.2f}/{acc['avg_counter'].avg:.2f}]-[%{acc['avg_counter_total'].avg:.2f}--{acc['avg_counter_pos'].avg:.2f}--{acc['avg_counter_neg'].avg:.2f}], loss:{loss.item():.2f}/{losses.avg:.2f},time consume:{time_counter.interval():.2f} s \r')
            self.optimizer_schedule.step()
            # 增加validation部分
            if validation:
                best_epoch=self.valid(_model,writer,epoch)
            # 2.2 保存好输出的结果，不要加到循环日志中去
            save_helper.save_epoch_pred(acc['epoch_acc_image'],
                                        'train_hardmine_%d_epoch_%d.txt' % (hard_mining_times, epoch))
            save_helper.save_epoch_model(hard_mining_times, epoch, 'train', acc, losses, _model)

            time_counter.addval(time.time(), key='training epoch end')
            self.log.info(('\ntrain epoch time consume:%.2f s' % (time_counter.key_interval(key_ed='training epoch end',
                                                                                            key_st='training epoch start'))))
#         return acc, losses

