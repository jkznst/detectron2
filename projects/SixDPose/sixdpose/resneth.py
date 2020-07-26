# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from detectron2.modeling import Backbone, BACKBONE_REGISTRY, ResNetBlockBase

__all__ = [
    "BottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "make_stage",
    "ResNet",
    "build_resneth_backbone",
    "build_crpnet_resneth_fpn_backbone",
]


# class ResNetBlockBase(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         """
#         The `__init__` method of any subclass should also contain these arguments.

#         Args:
#             in_channels (int):
#             out_channels (int):
#             stride (int):
#         """
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride

#     def freeze(self):
#         for p in self.parameters():
#             p.requires_grad = False
#         FrozenBatchNorm2d.convert_frozen_batchnorm(self)
#         return self


class BottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            stride_in_1x1 (bool): when stride==2, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)

        if (in_channels != out_channels) or (stride > 1):
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
    ):
        """
        Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
        """
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


def make_stage(block_class, num_blocks, first_stride, **kwargs):
    """
    Create a resnet stage by creating many blocks.

    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    use_dilation = kwargs["dilation"]
    for i in range(num_blocks):
        kwargs["dilation"] = 1 if not use_dilation else (i % 3 + 2)
        blocks.append(block_class(stride=first_stride if i == 0 else 1, **kwargs))
        kwargs["in_channels"] = kwargs["out_channels"]
    return blocks


class CSPStage(nn.Module):
    def __init__(self, block_class, num_blocks, first_stride, **kwargs):
        """
        Create a csp-resnet stage by creating many blocks.

        Args:
            block_class (class): a subclass of ResNetBlockBase
            num_blocks (int):
            first_stride (int): the stride of the first block. The other blocks will have stride=1.
                A `stride` argument will be passed to the block constructor.
            kwargs: other arguments passed to the block constructor.
        """
        super().__init__()
        self.blocks = []
        self.stride = first_stride
        use_dilation = kwargs["dilation"]
        in_channels = kwargs["in_channels"]
        out_channels = kwargs["out_channels"]
        self.out_channels = out_channels
        norm = kwargs["norm"]

        if first_stride == 1:   # stage 2
            self.pre_conv = Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, in_channels),
                )
            weight_init.c2_msra_fill(self.pre_conv)

            for i in range(num_blocks):
                kwargs["dilation"] = 1 if not use_dilation else (i % 3 + 2)
                kwargs["out_channels"] = out_channels // 2
                self.blocks.append(block_class(stride=1, **kwargs))
                kwargs["in_channels"] = kwargs["out_channels"]

            self.after_conv = Conv2d(
                out_channels // 2,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels // 2),
                )
            weight_init.c2_msra_fill(self.after_conv)

            self.csp_conv = Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels // 2),
            )
            weight_init.c2_msra_fill(self.csp_conv)
        elif first_stride > 1:  # stage 3 4 5
            self.transition_conv1 = Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, in_channels // 2),
            )
            weight_init.c2_msra_fill(self.transition_conv1)
            self.transition_conv2 = Conv2d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=3,
                stride=first_stride,
                padding=1,
                bias=False,
                norm=get_norm(norm, in_channels // 2),
            )
            weight_init.c2_msra_fill(self.transition_conv2)

            self.pre_conv = Conv2d(
                in_channels // 2,
                in_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, in_channels),
                )
            weight_init.c2_msra_fill(self.pre_conv)

            for i in range(num_blocks - 1):
                kwargs["dilation"] = 1 if not use_dilation else (i % 3 + 2)
                kwargs["out_channels"] = out_channels // 2
                self.blocks.append(block_class(stride=1, **kwargs))
                kwargs["in_channels"] = kwargs["out_channels"]

            self.after_conv = Conv2d(
                out_channels // 2,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels // 2),
                )
            weight_init.c2_msra_fill(self.after_conv)

            self.csp_conv = Conv2d(
                in_channels // 2,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm(norm, out_channels // 2),
            )
            weight_init.c2_msra_fill(self.csp_conv)
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        if self.stride == 1:
            out = self.pre_conv(x)
            out = F.relu_(out)
            
            for block in self.blocks:
                out = block(out)
            
            out = self.after_conv(x)
            out = F.relu_(out)

            csp = self.csp_conv(x)
            csp = F.relu_(csp)
            return torch.cat((out, csp), dim=1)
        elif self.stride > 1:
            x = self.transition_conv1(x)
            x = F.relu_(x)
            x = self.transition_conv2(x)
            x = F.relu_(x)

            csp = self.csp_conv(x)
            csp = F.relu_(csp)

            out = self.pre_conv(x)
            out = F.relu_(out)
            
            for block in self.blocks:
                out = block(out)

            out = self.after_conv(out)
            out = F.relu_(out)
            return torch.cat((out, csp), dim=1)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class BasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        # x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 2  # = stride 2 conv


class ResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class CSPResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[CSPStage]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(CSPResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, stage in enumerate(stages):     
            curr_channels = stage.out_channels
            name = "res" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * stage.stride)
            self._out_feature_channels[name] = stage.out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_resneth_backbone(cfg, input_shape):
    """
    Create a ResNet-h instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETH.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETH.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.RESNETH.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features        = cfg.MODEL.RESNETH.OUT_FEATURES
    depth               = cfg.MODEL.RESNETH.DEPTH
    num_groups          = cfg.MODEL.RESNETH.NUM_GROUPS
    # width_per_group     = cfg.MODEL.RESNETH.WIDTH_PER_GROUP
    bottleneck_channels = [32, 64, 128, 256]
    in_channels         = [64, 128, 256, 512]
    out_channels        = [128, 256, 512, 1024]
    stride_in_1x1       = cfg.MODEL.RESNETH.STRIDE_IN_1X1
    dilation_on_per_stage = cfg.MODEL.RESNETH.DILATION_ON_PER_STAGE
    deform_on_per_stage = cfg.MODEL.RESNETH.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETH.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETH.DEFORM_NUM_GROUPS
    # fmt: on
    # assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        # dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels[idx],
            "bottleneck_channels": bottleneck_channels[idx],
            "out_channels": out_channels[idx],
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation_on_per_stage[idx],
        }
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)


@BACKBONE_REGISTRY.register()
def build_cspresneth_backbone(cfg, input_shape):
    """
    Create a CSP-ResNet-h instance from config.

    Returns:
        CSPResNet: a :class:`CSPResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETH.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETH.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.RESNETH.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    out_features        = cfg.MODEL.RESNETH.OUT_FEATURES
    depth               = cfg.MODEL.RESNETH.DEPTH
    num_groups          = cfg.MODEL.RESNETH.NUM_GROUPS
    # width_per_group     = cfg.MODEL.RESNETH.WIDTH_PER_GROUP
    bottleneck_channels = [32, 64, 128, 256]
    in_channels         = [64, 128, 256, 512]
    out_channels        = [128, 256, 512, 1024]
    stride_in_1x1       = cfg.MODEL.RESNETH.STRIDE_IN_1X1
    dilation_on_per_stage = cfg.MODEL.RESNETH.DILATION_ON_PER_STAGE
    deform_on_per_stage = cfg.MODEL.RESNETH.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETH.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETH.DEFORM_NUM_GROUPS
    # fmt: on
    # assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        # dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels[idx],
            "bottleneck_channels": bottleneck_channels[idx],
            "out_channels": out_channels[idx],
            "num_groups": num_groups,
            "norm": norm,
            "stride_in_1x1": stride_in_1x1,
            "dilation": dilation_on_per_stage[idx],
        }
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = DeformBottleneckBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        csp_stage = CSPStage(**stage_kargs)

        if freeze_at >= stage_idx:
            csp_stage.freeze()
        stages.append(csp_stage)
    return CSPResNet(stem, stages, out_features=out_features)

class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]

class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu_(p6))
        return [p6, p7]


class FPN_resneth(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN_resneth, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx])) + 1
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s)) + 1): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** s

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            x[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

# class SSD_resneth(Backbone):
#     """
#     This module implements Feature Pyramid Network.
#     It creates pyramid features built on top of some input feature maps.
#     """

#     def __init__(
#         self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
#     ):
#         """
#         Args:
#             bottom_up (Backbone): module representing the bottom up subnetwork.
#                 Must be a subclass of :class:`Backbone`. The multi-scale feature
#                 maps generated by the bottom up network, and listed in `in_features`,
#                 are used to generate FPN levels.
#             in_features (list[str]): names of the input feature maps coming
#                 from the backbone to which FPN is attached. For example, if the
#                 backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
#                 of these may be used; order must be from high to low resolution.
#             out_channels (int): number of channels in the output feature maps.
#             norm (str): the normalization to use.
#             top_block (nn.Module or None): if provided, an extra operation will
#                 be performed on the output of the last (smallest resolution)
#                 FPN output, and the result will extend the result list. The top_block
#                 further downsamples the feature map. It must have an attribute
#                 "num_levels", meaning the number of extra FPN levels added by
#                 this block, and "in_feature", which is a string representing
#                 its input feature (e.g., p5).
#             fuse_type (str): types for fusing the top down features and the lateral
#                 ones. It can be "sum" (default), which sums up element-wise; or "avg",
#                 which takes the element-wise mean of the two.
#         """
#         super(SSD_resneth, self).__init__()
#         assert isinstance(bottom_up, Backbone)

#         # Feature map strides and channels from the bottom up network (e.g. ResNet)
#         input_shapes = bottom_up.output_shape()
#         in_strides = [input_shapes[f].stride for f in in_features]
#         in_channels = [input_shapes[f].channels for f in in_features]

#         _assert_strides_are_log2_contiguous(in_strides)
    
#         output_convs = []

#         use_bias = norm == ""
#         for idx, in_channel in enumerate(in_channels):
#             output_norm = get_norm(norm, out_channels)

#             output_conv = Conv2d(
#                 in_channel, out_channels, kernel_size=1, bias=use_bias, norm=output_norm
#             )
#             weight_init.c2_xavier_fill(output_conv)
#             stage = int(math.log2(in_strides[idx])) + 1
#             self.add_module("ssd_output{}".format(stage), output_conv)

#             output_convs.append(output_conv)
#         # Place convs into top-down order (from low to high resolution)
#         # to make the top-down computation in forward clearer.
#         self.output_convs = output_convs[::-1]
#         self.top_block = top_block
#         self.in_features = in_features
#         self.bottom_up = bottom_up
#         # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
#         self._out_feature_strides = {"p{}".format(int(math.log2(s)) + 1): s for s in in_strides}
#         # top block output feature maps.
#         if self.top_block is not None:
#             for s in range(stage, stage + self.top_block.num_levels):
#                 self._out_feature_strides["p{}".format(s + 1)] = 2 ** s

#         self._out_features = list(self._out_feature_strides.keys())
#         self._out_feature_channels = {k: out_channels for k in self._out_features}
#         self._size_divisibility = in_strides[-1]
#         assert fuse_type in {"avg", "sum"}
#         self._fuse_type = fuse_type

#     @property
#     def size_divisibility(self):
#         return self._size_divisibility

#     def forward(self, x):
#         """
#         Args:
#             input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
#                 feature map tensor for each feature level in high to low resolution order.

#         Returns:
#             dict[str->Tensor]:
#                 mapping from feature map name to FPN feature map tensor
#                 in high to low resolution order. Returned feature names follow the FPN
#                 paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
#                 ["p2", "p3", ..., "p6"].
#         """
#         # Reverse feature maps into top-down order (from low to high resolution)
#         bottom_up_features = self.bottom_up(x)
#         x = [bottom_up_features[f] for f in self.in_features[::-1]]
#         results = []
#         prev_features = x[0]
#         results.append(self.output_convs[0](prev_features))
#         for features, output_conv in zip(
#             x[1:], self.output_convs[1:]
#         ):
#             prev_features = features
#             results.insert(0, output_conv(prev_features))

#         if self.top_block is not None:
#             top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
#             if top_block_in_feature is None:
#                 top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
#             results.extend(self.top_block(top_block_in_feature))
#         assert len(self._out_features) == len(results)
#         return dict(zip(self._out_features, results))

#     def output_shape(self):
#         return {
#             name: ShapeSpec(
#                 channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
#             )
#             for name in self._out_features
#         }

class SSD_resneth(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(SSD_resneth, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
    
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s)) + 1): s for s in in_strides}
        # top block output feature maps.
        stage = 5
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** s

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {"p3": 256, "p4": 512, "p5": 1024, 
                            "p6": out_channels, "p7": out_channels}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = x[0]
        results.append(prev_features)
        for features in x[1:]:
            prev_features = features
            results.insert(0, prev_features)

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

@BACKBONE_REGISTRY.register()
def build_crpnet_resneth_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resneth_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN_resneth(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_crpnet_cspresneth_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_cspresneth_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN_resneth(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_crpnet_resneth_ssd_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resneth_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = SSD_resneth(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_crpnet_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    from detectron2.modeling import build_resnet_backbone, FPN
    
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["res5"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_hcrnet_resneth_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resneth_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN_resneth(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone