import random
import torch
import torchvision.transforms as transforms

"""transforms"""


# imagenet (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# blood cell image (0.678, 0.641, 0.660),(0.264, 0.263, 0.261)
# mitos aperio，(0.8383294001362148, 0.5878884803843312, 0.7093328145647311),(0.1400280688170852, 0.2305971217916802, 0.18305953619445997)
# mitos hamamatsu，(0.779, 0.492, 0.707), (0.148, 0.227, 0.155)

# 自定义transform
# 将图像的短边缩放到range内
class RandomScale(object):
    def __init__(self, shorter_side_range):
        self.shorter_side_range = shorter_side_range

    def __call__(self, img):
        shorter_side_scale = random.uniform(*self.shorter_side_range) / min(img.size)
        new_size = [round(shorter_side_scale * img.width), round(shorter_side_scale * img.height)]
        img = img.resize(new_size)
        return img


class MultiViewTenCrop(object):
    def __init__(self, multi_view, size=224, vertical_flip=False):
        self.multi_view = multi_view
        self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        img_list = []
        for view in self.multi_view:
            img_view = RandomScale((view, view))(img)
            img_ten = transforms.TenCrop(self.size)(img_view)
            img_list = img_list + list(img_ten)
        return transforms.Lambda(lambda crops: torch.stack(
            [transforms.Normalize((0.837, 0.584, 0.706), (0.141, 0.232, 0.184))(transforms.ToTensor()(crop)) for crop in
             crops]))(img_list)


def get_train_transforms(shorter_side_range=(224, 224), size=(224, 224)):
    return transforms.Compose([RandomScale(shorter_side_range=shorter_side_range),
                               transforms.RandomCrop(size=size),
                               transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_test_transforms(shorter_side_range=(224, 224), size=(224, 224)):
    return transforms.Compose([RandomScale(shorter_side_range=shorter_side_range),
                               transforms.RandomCrop(size=size),
                               transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
