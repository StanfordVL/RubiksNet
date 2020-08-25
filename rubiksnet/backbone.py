import torch
import math
import torch.nn as nn
from .shiftlib import *


__all__ = ["RubiksNetBackbone"]


def _skip_global_init(m):
    return hasattr(m, "skip_global_init") and m.skip_global_init


def conv2d_init(m):
    assert isinstance(m, nn.Conv2d)
    if _skip_global_init(m):
        return
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))


def conv_bn_init_module(net):
    assert isinstance(net, nn.Module)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            conv2d_init(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            norm_layer_init(m, weight_init=1.0)


def norm_layer_init(m, weight_init=1.0):
    assert isinstance(weight_init, (int, float))
    assert isinstance(m, (nn.BatchNorm2d, nn.GroupNorm))
    nn.init.constant_(m.weight, weight_init)
    nn.init.constant_(m.bias, 0)


def Conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def BN2d(planes, weight_init=1.0):
    bn = nn.BatchNorm2d(planes)
    norm_layer_init(bn, weight_init)
    return bn


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RubiksShiftBlock(nn.Module):
    def __init__(self, in_planes, out_planes, *, stride=1, parent):
        super().__init__()
        mid_planes = int(out_planes * parent.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = BN2d(in_planes)
        self.conv2 = Conv1x1(in_planes, mid_planes)
        self.bn2 = BN2d(mid_planes)
        self.as3 = RubiksShift2D(
            mid_planes,
            stride=stride,
            normalize_grad=parent.normalize_grad,
            quantize=parent.quantize,
            init_shift=parent.init_shift,
        )
        use_se = parent.use_se
        if use_se:
            if isinstance(use_se, bool):
                reduction = 12
            else:
                assert use_se > 2, ("SE reduction must > 2", use_se)
                reduction = use_se
            self.se = SELayer(mid_planes, reduction=reduction)
        else:
            self.se = None

        self.conv3 = Conv1x1(mid_planes, out_planes)

        if stride != 1 or in_planes != out_planes:
            self.shortcut = Conv1x1(in_planes, out_planes, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        shortcut is applied to out if num_channels change (i.e. at resnet group boundaries), else it's just x
        https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

        # pre-activation resnet version:
        def forward(self, x):
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
            return out
        """
        out = self.relu(self.bn1(x))
        if isinstance(self.shortcut, nn.Identity):
            shortcut = x
        else:
            shortcut = self.shortcut(out)
        out = self.conv2(out)
        out = self.relu(self.bn2(out))
        out = self.as3(out)
        if self.se:
            out = self.se(out)
        out = self.conv3(out)
        out += shortcut
        return out


class RubiksNetBackbone(nn.Module):
    def __init__(
        self,
        width,
        repeats,
        expansion=1,
        num_classes=1000,
        use_se=False,
        quantize=False,
        normalize_grad=True,
        init_shift="uniform",
    ):
        super().__init__()
        self.init_shift = init_shift
        self.width = width
        self.inplanes = width
        self.expansion = expansion
        self.use_se = use_se
        self.quantize = quantize
        self.normalize_grad = normalize_grad
        self.conv1 = Conv3x3(3, self.inplanes, stride=2)

        block = RubiksShiftBlock
        self.layer0 = self._make_layer(block, self.width, 1, stride=1)
        self.layer1 = self._make_layer(block, self.width, repeats[0], stride=2)
        self.layer2 = self._make_layer(block, 2 * self.width, repeats[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.width, repeats[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * self.width, repeats[3], stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.bn_last = BN2d(8 * self.width)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear((8 * self.width), num_classes)

        # ============= initialize parameters =============
        conv_bn_init_module(self)
        # according to FB large batch paper:
        self.fc.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, repeat, stride):
        """
        expansion_group: [first_expansion, (rest_expansion, repeat)]
        """
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, parent=self))
        self.inplanes = planes
        for _ in range(repeat - 1):
            layers.append(block(self.inplanes, planes, stride=1, parent=self,))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu(self.bn_last(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_optim_policy(self, shift_lr_mult=0.01):
        weight = []  # Conv2d and Linear only
        bias = []
        bn = []
        shift = []

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                ps = list(m.parameters())
                weight.append(ps[0])
                if len(ps) == 2:
                    bias.append(ps[1])
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn.extend(list(m.parameters()))
            elif isinstance(m, (RubiksShift2D, RubiksShiftBase)):
                shift.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. "
                        "Need to give it a learning policy".format(type(m))
                    )

        return [
            {"params": weight, "lr_mult": 1, "decay_mult": 1, "name": "weight"},
            {"params": bias, "lr_mult": 1, "decay_mult": 0, "name": "bias"},
            {"params": bn, "lr_mult": 1, "decay_mult": 0, "name": "bn"},
            {
                "params": shift,
                "lr_mult": shift_lr_mult,
                "decay_mult": 0,
                "name": "shift",
            },
        ]
