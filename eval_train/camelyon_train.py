from ..dataset import DynamicDataset,EvalDataset,ValidDataset
from ..eval_validate.camelyon_validate import Validate
import torchvision.models as models
import torchvision.transforms as transforms
# 血液细胞评测
from ..eval_train import basic_train
from ..model import camelyon_models

import torch,tqdm
from torch import nn
from torch import utils
from torch.autograd import Variable
import torch.optim as optim
from ..utils import accuracy
from ..utils import counter
from ..utils import logs
from ..utils import image_transform
from ..utils.checkpoint import Checkpointer
import time
import pdb
import os
import json
import torch.nn.functional as F
from ..dataset.sampler import RandomSampler
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

class Train(basic_train.BasicTrain):
    """包含每一轮train，返回每次batch_size的output
    """

    def __init__(self, config):
        """这里只解析和训练相关的参数，不对模型进行加载"""
        super(Train, self).__init__()
        self.config = config
        save_folder = config.get_config('base','save_folder')
        self.workspace=os.path.join(save_folder,'train')
        self.log = logs.Log(os.path.join(self.workspace, "log.txt"))
        writer_path = os.path.join(self.workspace,'visualize')
        os.system(f'mkdir -p {writer_path}')
        self.writer = SummaryWriter(writer_path)
        self.train_loader=self.load_data()
        self.valid =  Validate(self.config, self.workspace)
        self.ckpter = Checkpointer(self.workspace)

    def cfg(self, name):
        """获取配置简易方式"""
        return self.config.get_config('train', name)


    def load_data(self):
        _size = self.config.get_config('base', 'crop_size')
        train_transform = image_transform.get_train_transforms(shorter_side_range = (_size, _size), size = (_size, _size))
        if self.config.get_config('train','method','type') == 'base':
            train_dataset = EvalDataset(self.config.get_config('train', 'train_list'),
                                                      transform=train_transform,
                                                      tif_folder=self.config.get_config('base', 'train_tif_folder'),
                                                      patch_size=self.config.get_config('base', 'patch_size'))
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg('batch_size'),
                                               shuffle=True, num_workers=self.cfg('num_workers'))
        elif self.config.get_config('train','method','type') == 'on_the_fly':
            dynamicdata = DynamicDataset(tumor_list=self.config.get_config('train', 'tumor_list'),
                                                     normal_list=self.config.get_config('train', 'normal_list'),
                                                     transform=train_transform,
                                                     data_size=self.config.get_config('train','data_size'),
                                                     replacement=self.config.get_config('train','replacement'),
                                                     tif_folder=self.config.get_config('base', 'train_tif_folder'),
                                                     patch_size=self.config.get_config('base', 'patch_size'))
            # self.log.info("update dataset")
            sampler = RandomSampler(dynamicdata,self.config.get_config('train','method','datasize'))
            return torch.utils.data.DataLoader(dynamicdata, batch_size=self.cfg('batch_size'),
                                               sampler=sampler, num_workers=self.cfg('num_workers'))


    def init_optimizer(self, _model):
        _params = self.cfg('params')
        self.optimizer = optim.SGD(_model.parameters(), lr=_params['lr_start'], momentum=_params['momentum'],
                                         weight_decay=_params['weight_decay'])
        self.optimizer_schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=_params['lr_decay_epoch'],
                                                            gamma=_params['lr_decay_factor'], last_epoch=-1)

    def validation(self,_model,epoch):
        return self.valid.run(_model,epoch)


    def train_epoch(self, _model,epoch,  criterion):
        _model.train()
        acc = {'correct_pos': counter.Counter(), 'total_pos': counter.Counter(),
               'correct_neg': counter.Counter(), 'total_neg': counter.Counter()}
        losses = counter.Counter()
        time_counter = counter.Counter()
        time_counter.addval(time.time(), key='training epoch start')
        for i, data in enumerate(tqdm.tqdm(self.load_data(),dynamic_ncols=True, leave=False), 0):
            _input, _labels, path_list = data
            if torch.cuda.is_available():
                _input = Variable(_input.type(torch.cuda.FloatTensor))
            else:
                _input = Variable(_input.type(torch.FloatTensor))
            _output = _model(_input).squeeze().cpu()
            #             pdb.set_trace()
            loss = criterion(_output, _labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #             pdb.set_trace()
            _output = F.softmax(_output)[:, 1].detach()
            correct_pos, total_pos, correct_neg, total_neg = accuracy.acc_binary_class(_output, _labels, 0.5)
            acc['correct_pos'].addval(correct_pos)
            acc['total_pos'].addval(total_pos)
            acc['correct_neg'].addval(correct_neg)
            acc['total_neg'].addval(total_neg)
            losses.addval(loss.item(), len(_output))
        TP = acc['correct_pos'].sum
        total_pos = acc['total_neg'].sum
        TN = acc['correct_neg'].sum
        total_neg = acc['total_pos'].sum
        total = total_pos + total_neg
        time_counter.addval(time.time())
        total_neg = acc['total_pos'].sum
        total = total_pos + total_neg
        total_acc = (TP + TN) / total
        pos_acc = TP / total_pos
        neg_acc = TN / total_neg
        self.log.info(
            'train new epoch:%d,batch iter:%d/%d, lr:%.5f, [total:%.2f-pos:%.2f-neg:%.2f], loss:%.2f/%.2f,time consume:%.2f s' % (
                epoch, i, len(self.train_loader),
                self.optimizer.state_dict()['param_groups'][0]['lr'],
                total_acc, pos_acc, neg_acc,
                loss.item(),
                losses.avg,
                time_counter.interval()), '\r')
        return total_acc,pos_acc,neg_acc, losses.avg

    def run(self, _model):
        criterion = nn.CrossEntropyLoss()
        train_epoch_stop = self.cfg('total_epoch')
        ckpt = self.ckpter.load(self.cfg("resume", "start_epoch"))
        if ckpt[0]:
            _model.load_state_dict(ckpt[0])
            self.optimizer=self.optimizer.load_state_dict(ckpt[1])
            train_epoch_start=ckpt[2]

        else:
            self.init_optimizer(_model)
            train_epoch_start = 0

        for epoch in range(train_epoch_start, train_epoch_stop):
            total_acc,pos_acc,neg_acc, loss=self.train_epoch(_model,epoch, criterion)
            self.writer.add_scalar('acc_total in train', total_acc, epoch)
            self.writer.add_scalar('acc_pos in train', pos_acc, epoch)
            self.writer.add_scalar('acc_neg in train', neg_acc, epoch)
            self.writer.add_scalar('loss in train', loss, epoch)
            state_dict = {
                "net": _model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "last_epoch": epoch,
            }
            self.ckpter.save(epoch, state_dict, total_acc)
            result = self.validation(_model, epoch)
            self.writer.add_scalar('acc_total in valid', result[0], epoch)
            self.writer.add_scalar('acc_pos in valid', result[1], epoch)
            self.writer.add_scalar('acc_neg in valid', result[2], epoch)
            self.writer.add_scalar('loss in valid', result[3], epoch)
            # 2.2 保存好输出的结果，不要加到循环日志中去
