from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision as V
from torch import Tensor
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.resnet import ResNet, model_urls


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    # copied from https://github.com/pytorch/vision/blob/095cabb76cdcd4763bad629481d189b91e3df42c/torchvision/models/resnet.py
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    # copied from https://github.com/pytorch/vision/blob/095cabb76cdcd4763bad629481d189b91e3df42c/torchvision/models/resnet.py
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # copied from https://github.com/pytorch/vision/blob/095cabb76cdcd4763bad629481d189b91e3df42c/torchvision/models/resnet.py
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PartialResNet(ResNet):
    def __init__(self, pretrained=True):
        super().__init__(V.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1_000)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)
        self.eval()

    def features(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
