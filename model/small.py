import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import pickle
import numpy as np

class SmallNet(nn.Module):

    def __init__(self, input_channel=3, num_classes=10):
        super(SmallNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),            
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16384, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SmallNet2(nn.Module):

    def __init__(self, input_channel=3, num_classes=10):
        super(SmallNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16384, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SmallNet3(nn.Module):

    def __init__(self, input_channel=3, num_classes=10):
        super(SmallNet3, self).__init__()
        self.features = nn.Sequential(
        )
        self.classifier = nn.Sequential(
            nn.Linear(9*32*32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
