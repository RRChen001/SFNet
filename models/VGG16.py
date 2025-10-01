import torch
import torchvision
from torchvision import models
from collections import namedtuple

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1_1 = torch.nn.Sequential()
        self.slice2_1 = torch.nn.Sequential()
        self.slice3_1 = torch.nn.Sequential()
        self.slice4_1 = torch.nn.Sequential()
        self.slice5_1 = torch.nn.Sequential()
        self.slice1_2 = torch.nn.Sequential()
        self.slice2_2 = torch.nn.Sequential()
        self.slice3_2 = torch.nn.Sequential()
        self.slice4_2 = torch.nn.Sequential()
        self.slice5_2 = torch.nn.Sequential()
        for x in range(2):
            self.slice1_1.add_module(str(x), vgg_pretrained_features[x])     #将卷积操作放在对应的顺序
        for x in range(2, 4):
            self.slice1_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 7):
            self.slice2_1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 9):
            self.slice2_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 12):
            self.slice3_1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 14):
            self.slice3_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 19):
            self.slice4_1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 21):
            self.slice4_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 26):
            self.slice5_1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(26, 28):
            self.slice5_2.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1_1(X)
        h_relu1_1 = h
        h = self.slice1_2(h)
        h_relu1_2 = h
        h = self.slice2_1(h)
        h_relu2_1 = h
        h = self.slice2_2(h)
        h_relu2_2 = h
        h = self.slice3_1(h)
        h_relu3_1 = h
        h = self.slice3_2(h)
        h_relu3_2 = h
        h = self.slice4_1(h)
        h_relu4_1 = h
        h = self.slice4_2(h)
        h_relu4_2 = h
        h = self.slice5_1(h)
        h_relu5_1 = h
        h = self.slice5_2(h)
        h_relu5_2 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1', 'relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2'])
        out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1, h_relu1_2, h_relu2_2, h_relu3_2, h_relu4_2, h_relu5_2)
        return out