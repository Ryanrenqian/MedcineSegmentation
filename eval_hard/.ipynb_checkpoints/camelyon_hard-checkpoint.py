import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from  torch.autograd import Variable
import time

from basic import BasicHard
from basic.utils import image_transform
from basic import camelyon_data
from basic.utils import logs
from basic.utils import counter
from basic.utils import accuracy
from basic.model import camelyon_models

import json
import glob
import numpy as np
import random
from  random import shuffle
import time
import pdb


class Hard(BasicHard):
    """模型性能的hard部份
    内部不加载模型，只通过load model处理"""

    def __init__(self, config):
        super(Hard, self).__init__()
        #         pdb.set_trace()
        self.config = config
        save_folder = os.path.join(self.config.get_config('base', 'save_folder'),
                                   self.config.get_config('base', 'last_run_date'))
        self.log = logs.Log(os.path.join(save_folder, "log.txt"))
        # dataloader
        self.hard_loader = None

        self.after_model_output = getattr(camelyon_models, 'after_model_output')

    def cfg(self, name):
        """获取配置简易方式"""
        return self.config.get_config('hard', name)


    def load_data(self):
        hard_transform = image_transform.get_test_transforms()
        hard_dataset = camelyon_data.HardDataset(self.config.get_config('base', 'hard_mining_list'),
                                                 transform=hard_transform,
                                                 tif_folder=self.config.get_config('base', 'train_tif_folder'),
                                                 patch_size=self.config.get_config('base', 'patch_size'))

        return torch.utils.data.DataLoader(hard_dataset, batch_size=self.cfg('batch_size'),
                                           shuffle=True, num_workers=self.cfg('num_workers'))

    def hard(self, _model, epoch, hard_mining_times, save_helper):
        _model.eval()
        
        time_counter = counter.Counter()
        time_counter.addval(time.time(), key='hard epoch start')
        # 1.生成单张的slide列表，按slide遍历，得到中间结果
        # 1.1 读取数据
        print('start loading hard list')
        load_time = time.time()
        _f = open(self.config.get_config('base', 'hard_mining_list'), 'r')
        content = _f.read()
        print('read txt from  disk complete!', time.time() - load_time)
        patch_dict = json.loads(content)
        print('loading hard json list complete! time:%0.4f  total patch:%d' % ((time.time() - load_time), len(patch_dict)))
        # 1.1 将数据按照slide的维度划分, {slide0:{xxx:0}}
        slide_patch_dict = {}
        print('start split patch list')
        patch_time = time.time()
        for patch_name in patch_dict.keys():
            slide_name = patch_name.split('.tif_')[0] + '.tif'
            if slide_name in slide_patch_dict.keys():
                slide_patch_dict[slide_name][patch_name] = patch_dict[patch_name]
            else:
                slide_patch_dict[slide_name] = {}
                slide_patch_dict[slide_name][patch_name] = patch_dict[patch_name]
        print('split patch list complete')
        for k in slide_patch_dict.keys():
            print('slide:%s patch:%d' % (k, len(slide_patch_dict[k])))

        # 2.以slide_name为维度去hard
        slide_name_list = list(slide_patch_dict.keys())
        slide_name_list.sort()
        
        start_index = self.config.get_config('hard', 'start_index')
        end_index = self.config.get_config('hard', 'end_index')
        end_index = end_index if end_index!=0 else len(slide_name_list)
        for slide_index in range(start_index, end_index):
            slide_name = slide_name_list[slide_index]
            hard_transform = image_transform.get_test_transforms()
            hard_dataset = camelyon_data.HardDataset(slide_name, slide_patch_dict[slide_name],
                                                     transform=hard_transform, tif_folder=self.config.get_config('base', 'train_tif_folder'))
            dataloader = torch.utils.data.DataLoader(hard_dataset, batch_size=self.cfg('batch_size'),
                                                     shuffle=False, num_workers=self.cfg('num_workers'))

            # 2.1 遍历该slide的patch
            acc = {'avg_counter_05': counter.Counter(), 'avg_counter_08': counter.Counter(),
                   'avg_counter_09': counter.Counter(),
                   'avg_counter': counter.Counter(), 'epoch_acc_image': []}
            for i, data in enumerate(dataloader, 0):
                hard_input, batch_labels, path_list = data
                if torch.cuda.is_available():
                    hard_input = Variable(hard_input.type(torch.cuda.FloatTensor))
                else:
                    hard_input = Variable(hard_input.type(torch.FloatTensor))
                #             hard_input = Variable(hard_input.type(torch.FloatTensor))
                hard_output = _model(hard_input)
                
                hard_output = self.after_model_output(hard_output, self.config)
                hard_output = hard_output.cpu()
                batch_labels = Variable(batch_labels.type(torch.FloatTensor))

                # 结果处理，计算acc
                acc_batch_05 = accuracy.acc_two_class(hard_output.cpu(), batch_labels, 0.5)
                acc_batch_08 = accuracy.acc_two_class(hard_output.cpu(), batch_labels, 0.8)
                acc_batch_09 = accuracy.acc_two_class(hard_output.cpu(), batch_labels, 0.9)
                acc_batch = acc_batch_08
                acc['avg_counter_05'].addval(acc_batch_05)
                acc['avg_counter_08'].addval(acc_batch_08)
                acc['avg_counter_09'].addval(acc_batch_09)
                acc['avg_counter'].addval(acc_batch)

                acc_image_list = accuracy.acc_two_class_image(hard_output.cpu(), batch_labels, path_list, 0.8)
                acc['epoch_acc_image'] = acc['epoch_acc_image'] + acc_image_list
                time_counter.addval(time.time())
                self.log.info(
                    'hard epoch slide:%d/%d %s,batch iter:%d/%d,acc-iter/avg:[%.2f/%.2f]-[%.2f--%.2f--%.2f], time consume:%.2f s' % (
                        epoch, self.config.get_config('train', 'total_epoch'), slide_name, i, len(dataloader),
                        acc_batch,
                        acc['avg_counter'].avg,
                        acc['avg_counter_05'].avg, acc['avg_counter_08'].avg, acc['avg_counter_09'].avg,
                        time_counter.interval()), end='\r')
                pass

            # 2.2 保存好输出的结果，不加到循环日志中去
            save_helper.save_epoch_pred(acc['epoch_acc_image'],
                                        'hard_hardmine_%d_epoch_%d_slide_%s.txt' % (
                                        hard_mining_times, epoch, slide_name))

            time_counter.addval(time.time(), key='hard epoch end')
            self.log.info(
                ('\nhard epoch slide time consume:%.2f s' % (time_counter.key_interval(key_ed='hard epoch end',
                                                                                       key_st='hard epoch start'))))
            hard_dataset.slide.close()
        
        
        