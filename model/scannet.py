#from  basic.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys

class Scannet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, pretrained=True):
        super(Scannet, self).__init__()
        vgg =models.vgg16(pretrained)
        features = list(vgg.features.children())

        # Set padding=0 in features
        for layer in features:
            layer.padding = (0, 0)
        self.features = nn.Sequential(*features)
        # repalce the FC layer of vgg 166
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, stride=1, padding=0),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=1024, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

