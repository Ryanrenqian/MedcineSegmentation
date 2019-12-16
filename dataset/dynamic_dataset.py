from  PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import torch.utils.data as data
import json
import os
import glob
import openslide
import os
import pdb
import time
import numpy as np
class ListDataset(data.Dataset):
    def __init__(self, list_file,transform=None,all_class=None, tif_folder='/root/workspace/dataset/CAMELYON16/training/*',
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
        with open(list_file,'r')as f:
            self.patch_name_list=f.readlines()
        self.all_class=all_class
        self.patch_size = patch_size
        self.transform = transform
        # 添加所有的slide缓存，从缓存中取数据
        self.slide_dict = {}
        for tif in tif_list:
            basename = os.path.basename(tif)
            self.slide_dict[basename] = tif

    def __getitem__(self, index):
        if self.all_class==None:
            patch_name,class_id = self.patch_name_list[index].rstrip().split()
        else:
            patch_name = self.patch_name_list[index].rstrip()
            class_id = self.all_class
        slide_name = patch_name.split('.tif_')[0] + '.tif'
        slide = openslide.OpenSlide(self.slide_dict[slide_name])  # 直接在这里使用对速度没有明显影响，但slide的缓存会较少很多
        _x, _y = patch_name.split('.tif_')[1].split('_')
        _x, _y = int(_x), int(_y)
        input_img = None
        # print(_x,_y,slide_name)
        try:
            img = slide.read_region((_x, _y), 0, [self.patch_size, self.patch_size]).convert('RGB')
            input_img = self.transform(img)
        except Exception as e:
            print(str(e))
            print('Image error:%s/n/n' % patch_name)
            input_img, class_id, patch_name = self.__getitem__(0)
        # input_img = input_img.cuda()
        return input_img, class_id, patch_name

    def __len__(self):
        return len(self.patch_name_list)


class DynamicDataset():
    def __init__(self,normal_list,tumor_list,data_size,transform,patch_size,replacement=False, tif_folder='/root/workspace/dataset/CAMELYON16/training/*'):
        self.tumor = ListDataset(list_file=tumor_list,
                                 tif_folder=tif_folder,
                                 transform=transform,
                                 all_class=1,
                                 patch_size=patch_size)
        self.normal = ListDataset(list_file=normal_list,
                                  tif_folder=tif_folder,
                                  transform=transform,
                                  all_class=0,
                                  patch_size=patch_size)
        self.data_size =data_size
        self.replacement=replacement

    @property
    def data(self):
        '''

        :return: Tumor + Normal dataset
        '''
        return data.ConcatDataset([self.tumor,self.normal])

    def  __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.tumor)+len(self.normal)

    @property
    def shape(self):
        '''

        :return: tumor size, normal size
        '''
        return (len(self.tumor),len(self.normal))

