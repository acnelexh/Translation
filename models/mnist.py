import math
import torch
from torch import nn

import torchvision.models as models


__all__ = [
    'ResNet50',
    'VGG16'
]

def _backbone_init(backbone: nn.Module):
    for layer in backbone.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)
        if isinstance(layer, nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, 1.0)
                torch.nn.init.constant_(layer.bias, 0.0)

def _head_init(head: nn.Module):
    for layer in head.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

# for resnet50 FPN
def ResNet50():
    model = ResNet50()
    return model

class ResNet50(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        resnet50 = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        self.head = nn.Linear(2048, num_class)
        for params in self.backbone.parameters():
            params.requires_grad_()
        for params in self.head.parameters():
            params.requires_grad_()
        self._unfreeze_n_initialize()

    def _unfreeze_n_initialize(self):
        for params in self.backbone.parameters():
            params.requires_grad_()
        for params in self.head.parameters():
            params.requires_grad_()
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Reinitiailize all params.
        Conv2d: xavier_uniform
        BatchNorm2d: 1 & 0
        Linear: normal & constant
        '''
        _backbone_init(self.backbone)
        _head_init(self.head)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def VGG16(a):
    model = VGG16()
    return model

class VGG16(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        VGG16 = models.vgg16(pretrained=False)
        self.backbone = nn.Sequential(*list(VGG16.children())[:-1])
        print(self.backbone)
        test_output_shape(self.backbone) # N x 3 x 32 x 32 => N x 512 x 7 x 7
    
    def forward(self, x):
        x = self.backbone(x)
        return x

def test_output_shape(model):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    output = model(dummy)
    print(output.shape)

if __name__ == "__main__":
    num_class = 10
    model = ResNet50(num_class)