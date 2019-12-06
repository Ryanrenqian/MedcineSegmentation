from  PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import json
import os
import glob
import openslide
import os
import pdb
import time
import numpy as np
from basic.data.dynamic_dataset import *
class EvalDataset(data.Dataset):
    def __init__(self, _patch_list_txt,  tif_folder='/root/workspace/dataset/CAMELYON16/training/*',
                 patch_size=256):
        """
        _patch_list_txt：
        {'xxx.tif_x_y': 0,
         'xxx.tif_x_y': 1,
        } 其中x_y指的是level0上的值
        :param _patch_list_txt:
        :param transform:
        """
        tif_list = glob.glob(os.path.join(tif_folder, '*.tif'))
        tif_list.sort()
        _f = open(_patch_list_txt, 'r')
        self.patch_dict = json.loads(_f.read())
        self.patch_name_list = [crop for crop in self.patch_dict.keys()]
        self.patch_size = patch_size
        # 添加所有的slide缓存，从缓存中取数据
        self.slide_dict = {}
        for tif in tif_list:
            basename = os.path.basename(tif)
            self.slide_dict[basename] = tif

    def __getitem__(self, index):
        patch_name = self.patch_name_list[index]
        class_id = self.patch_dict[patch_name]
        slide_name = patch_name.split('.tif_')[0] + '.tif'
        slide = openslide.OpenSlide(self.slide_dict[slide_name])  # 直接在这里使用对速度没有明显影响，但slide的缓存会较少很多
        _x, _y = patch_name.split('.tif_')[1].split('_')
        _x, _y = int(_x), int(_y)

        input_img = None
        try:
            img = slide.read_region((_x, _y), 0, [self.patch_size, self.patch_size]).convert(
                'RGB')
            input_img = self.transform(img)
        except Exception as e:
            print(str(e))
            print('Image error:%s/n/n' % patch_name)
            input_img, class_id, patch_name = self.__getitem__(0)
        # input_img = input_img.cuda()
        return input_img, class_id, patch_name

    def __len__(self):
        return len(self.patch_name_list)


class TestDataset(data.Dataset):
    def __init__(self, slide_name, patch_dict, transform=None,
                 tif_folder='/root/workspace/dataset/CAMELYON16/testing/images', patch_size=256):
        """
        与training的区别在于：为了提升速度，test数据集是逐张slide的patch统一处理
        patch_dict：
        {'xxx.tif_x_y': 0,
         'xxx.tif_x_y': 1,
        } 其中x_y指的是level0上的值
        :param slide_name:
        :param transform:
        """
        slide_path = os.path.join(tif_folder, slide_name)
        print('handle slide: %s' % slide_path)
        self.slide = openslide.OpenSlide(slide_path)
        self.patch_dict = patch_dict
        self.patch_list = list(patch_dict.keys())

        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, index):
        patch_name = self.patch_list[index]
        class_id = self.patch_dict[patch_name]
        _x, _y = patch_name.split('.tif_')[1].split('_')
        _x, _y = int(_x), int(_y)
        input_img = None
        try:
            img = self.slide.read_region((_x, _y), 0, [self.patch_size, self.patch_size]).convert(
                'RGB')
            input_img = self.transform(img)
        except Exception as e:
            print(str(e))
            print('Image error:%s/n/n' % patch_name)
            input_img, class_id, patch_name = self.__getitem__(0)
        # input_img = input_img.cuda()
        return input_img, class_id, patch_name

    def __len__(self):
        return len(self.patch_list)


class HardDataset(data.Dataset):
    def __init__(self, slide_name, patch_dict, transform=None,
                 tif_folder='/root/workspace/dataset/CAMELYON16/training/*', patch_size=256):
        """
        与training的区别在于：为了提升速度，hard数据集是逐张slide的patch统一处理
        patch_dict：
        {'xxx.tif_x_y': 0,
         'xxx.tif_x_y': 1,
        } 其中x_y指的是level0上的值
        :param slide_name:
        :param transform:
        """
        slide_path = glob.glob(os.path.join(tif_folder, slide_name))[0]
        print('handle slide: %s' % slide_path)
        self.slide = openslide.OpenSlide(slide_path)
        self.patch_dict = patch_dict
        self.patch_list = list(patch_dict.keys())

        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, index):
        patch_name = self.patch_list[index]
        class_id = self.patch_dict[patch_name]
        _x, _y = patch_name.split('.tif_')[1].split('_')
        _x, _y = int(_x), int(_y)
        input_img = None
        try:
            img = self.slide.read_region((_x, _y), 0, [self.patch_size, self.patch_size]).convert(
                'RGB')
            input_img = self.transform(img)
        except Exception as e:
            print(str(e))
            print('Image error:%s/n/n' % patch_name)
            input_img, class_id, patch_name = self.__getitem__(0)
        # input_img = input_img.cuda()
        return input_img, class_id, patch_name

    def __len__(self):
        return len(self.patch_list)
