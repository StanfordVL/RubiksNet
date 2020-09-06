import os

import torch.nn as nn
from .shiftlib import *
from .attention_shift import AttentionShift
from .utils import *
from .backbone import *


__all__ = ["RubiksNet"]


class RubiksNet(nn.Module):
    def __init__(self, tier, num_classes, num_frames=8, variant="rubiks3d"):
        super().__init__()
        assert tier in ["tiny", "small", "medium", "large"]
        assert variant in ["rubiks3d", "rubiks3d-aq"]

        self.num_frames = num_frames
        self.tier = tier
        self.variant = variant

        print(
            f'Initializing RubiksNet-{tier.capitalize()} variant "{variant}". '
            f"num_frames={self.num_frames}"
        )

        if tier == "tiny":
            self.backbone = RubiksNetBackbone(
                width=54, repeats=[3, 4, 6, 3], num_classes=num_classes, use_se=False,
            )
        elif tier == "small":
            self.backbone = RubiksNetBackbone(
                width=72, repeats=[3, 4, 6, 3], num_classes=num_classes, use_se=True,
            )
        elif tier == "medium":
            self.backbone = RubiksNetBackbone(
                width=72, repeats=[3, 4, 23, 3], num_classes=num_classes, use_se=False,
            )
        elif tier == "large":
            self.backbone = RubiksNetBackbone(
                width=72, repeats=[3, 8, 36, 3], num_classes=num_classes, use_se=False
            )
        else:
            raise NotImplementedError(f"Unknown tier {tier}")

        self._prepare_backbone()
        self.feature_dim = getattr(self.backbone, self.backbone.last_layer_name).in_features
        setattr(self.backbone, self.backbone.last_layer_name, nn.Identity())
        self.new_fc = nn.Linear(self.feature_dim, num_classes)

    @classmethod
    def load_pretrained(cls, ckpt_path):
        ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu")
        net = cls(
            tier=ckpt["tier"],
            num_classes=ckpt["num_classes"],
            num_frames=ckpt["num_frames"],
            variant=ckpt["variant"],
        )
        net.load_state_dict(ckpt['model'])
        return net

    def replace_new_fc(self, num_classes):
        self.new_fc = nn.Linear(self.feature_dim, num_classes)

    def _prepare_backbone(self):
        num_frames = self.num_frames
        net = self.backbone

        if self.variant == "rubiks3d-aq":
            print("RubiksNet attention quantized shift (before block.conv2)")

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    shift_layer = AttentionShift(num_frames)
                    b.conv2 = nn.Sequential(shift_layer, b.conv2)
                return nn.Sequential(*blocks)

        elif self.variant == "rubiks3d":
            print("=> RubiksShift3D")

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    b.as3 = _Rubiks3DWrap(b.as3, n_segment=this_segment)
                return nn.Sequential(*blocks)

        else:
            raise NotImplementedError(f"Unknown variant: {self.variant}")

        if hasattr(net, "layer0"):
            net.layer0 = make_block_temporal(net.layer0, num_frames)
        net.layer1 = make_block_temporal(net.layer1, num_frames)
        net.layer2 = make_block_temporal(net.layer2, num_frames)
        net.layer3 = make_block_temporal(net.layer3, num_frames)
        net.layer4 = make_block_temporal(net.layer4, num_frames)

        if self.variant == "rubiks3d-aq":
            print("=> setup forward prop for temporal attention shift")
            x = torch.zeros((8, 3, 224, 224))
            net.eval().cuda()(x.cuda())
            net.train()

        net.last_layer_name = "fc"
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        net.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input):
        base_out = self.backbone(input.view((-1, 3) + input.size()[-2:]))
        base_out = self.new_fc(base_out)
        base_out = base_out.view((-1, self.num_frames) + base_out.size()[1:])
        output = base_out.mean(dim=1, keepdim=True)
        return output.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224


class _Rubiks3DWrap(nn.Module):
    def __init__(self, rubiks2d, n_segment=8):
        super().__init__()
        assert isinstance(rubiks2d, RubiksShift2D)
        self.rubiks3d = RubiksShift3D(
            rubiks2d.num_channels,
            stride=(1, *make_tuple(rubiks2d.stride, 2)),
            padding=(0, *make_tuple(rubiks2d.padding, 2)),
        )
        self.n_segment = n_segment

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        out = self.rubiks3d(x)
        n, t, c, h, w = out.size()
        return out.view(n * t, c, h, w)
