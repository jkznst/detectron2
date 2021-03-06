# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling import Backbone, BACKBONE_REGISTRY, build_resnet_backbone

__all__ = ["build_resnet_fpg_backbone", "FPG"]


class FPG(Backbone):
    """
    This module implements Feature Pyramid Grids.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, num_pathways=1, norm="", fuse_type="sum"
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
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPG, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        p1_lateral_convs = []
        p1_sameup_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            p1_lateral_norm = get_norm(norm, out_channels)
            p1_sameup_norm = get_norm(norm, out_channels)

            p1_lateral_conv = Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                bias=use_bias, 
                norm=p1_lateral_norm,
                activation=F.relu
            )
            p1_sameup_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias,
                norm=p1_sameup_norm,
                activation=F.relu
            )
            weight_init.c2_xavier_fill(p1_lateral_conv)
            weight_init.c2_xavier_fill(p1_sameup_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpg_p1_lateral{}".format(stage), p1_lateral_conv)
            self.add_module("fpg_p1_sameup{}".format(stage), p1_sameup_conv)

            p1_lateral_convs.append(p1_lateral_conv)
            p1_sameup_convs.append(p1_sameup_conv)
        
        self.p1_lateral_convs = p1_lateral_convs
        self.p1_sameup_convs = p1_sameup_convs
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        self._out_feature_strides["p{}".format(stage + 1)] = 2 ** (stage + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1] * 2
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type
        self.num_pathways = num_pathways

        for idx_pathway in range(num_pathways - 1):
            lateral_convs = []
            sameup_convs = []
            acrossdown_convs = []
            skip_connections = []

            for idx, out_feature in enumerate(self._out_features):
                lateral_norm = get_norm(norm, out_channels)
                sameup_norm = get_norm(norm, out_channels)
                acrossdown_norm = get_norm(norm, out_channels)
                # skip_norm = get_norm(norm, out_channels)

                if idx + idx_pathway < 3:
                    lateral_conv = nn.Identity()
                    sameup_conv = None
                    acrossdown_conv = None
                    skip_connection = None
                else:
                    lateral_conv = Conv2d(
                        out_channels, 
                        out_channels, 
                        kernel_size=1, 
                        bias=use_bias, 
                        norm=lateral_norm,
                        activation=F.relu
                    )
                    sameup_conv = Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                        norm=sameup_norm,
                        activation=F.relu
                    )
                    acrossdown_conv = Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                        norm=acrossdown_norm,
                        activation=F.relu
                    )
                    # skip_connection = Conv2d(
                    #     out_channels, 
                    #     out_channels, 
                    #     kernel_size=1, 
                    #     bias=use_bias, 
                    #     norm=skip_norm,
                    #     activation=F.relu
                    # )
                    skip_connection = True
                    weight_init.c2_xavier_fill(lateral_conv)
                    weight_init.c2_xavier_fill(sameup_conv)
                    weight_init.c2_xavier_fill(acrossdown_conv)
                    # weight_init.c2_xavier_fill(skip_connection)
                stage = int(idx + 2)
                if idx == 0:
                    self.add_module("fpg_p{}_lateral{}".format(idx_pathway + 2, stage), lateral_conv)
                    # if skip_connection is not None:
                    #     self.add_module("fpg_p{}_skipconnection{}".format(idx_pathway + 2, stage), skip_connection)

                    lateral_convs.append(lateral_conv)
                    skip_connections.append(skip_connection)
                else:
                    self.add_module("fpg_p{}_lateral{}".format(idx_pathway + 2, stage), lateral_conv)
                    if sameup_conv is not None:
                        self.add_module("fpg_p{}_sameup{}".format(idx_pathway + 2, stage), sameup_conv) 
                    if acrossdown_conv is not None:
                        self.add_module("fpg_p{}_acrossdown{}".format(idx_pathway + 2, stage), acrossdown_conv) 
                    # if skip_connection is not None:
                    #     self.add_module("fpg_p{}_skipconnection{}".format(idx_pathway + 2, stage), skip_connection)

                    lateral_convs.append(lateral_conv)
                    skip_connections.append(skip_connection)
                    sameup_convs.append(sameup_conv)
                    acrossdown_convs.append(acrossdown_conv)

            lateral_convs_name = "p{}_lateral_convs".format(idx_pathway + 2)
            sameup_convs_name = "p{}_sameup_convs".format(idx_pathway + 2)
            acrossdown_convs_name = "p{}_acrossdown_convs".format(idx_pathway + 2)
            skip_connections_name = "p{}_skip_connections".format(idx_pathway + 2)
            setattr(self, lateral_convs_name, lateral_convs)
            setattr(self, sameup_convs_name, sameup_convs)
            setattr(self, acrossdown_convs_name, acrossdown_convs)
            setattr(self, skip_connections_name, skip_connections)

        # 3x3 vanilla output convolutions
        output_convs = []
        for idx, out_feature in enumerate(self._out_features):
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias
            )
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("fpg_output{}".format(idx + 2), output_conv)
            output_convs.append(output_conv)
        self.output_convs = output_convs

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
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features]
        p1_features = []
        prev_features = self.p1_lateral_convs[0](x[0])
        p1_features.append(prev_features)
        for features, p1_lateral_conv, p1_sameup_conv in zip(
            x[1:], self.p1_lateral_convs[1:], self.p1_sameup_convs[0:-1]
        ):
            bottom_up_features = p1_sameup_conv(prev_features)
            lateral_features = p1_lateral_conv(features)
            prev_features = lateral_features + bottom_up_features
            if self._fuse_type == "avg":
                prev_features /= 2
            p1_features.append(prev_features)
        p1_features.append(self.p1_sameup_convs[-1](prev_features)) # [p1_2,p1_3,p1_4,p1_5,p1_6]
        
        pathway_features = [p1_features]
        for idx_pathway in range(self.num_pathways - 1):
            lateral_convs = getattr(self, "p{}_lateral_convs".format(idx_pathway + 2))
            sameup_convs = getattr(self, "p{}_sameup_convs".format(idx_pathway + 2))
            acrossdown_convs = getattr(self, "p{}_acrossdown_convs".format(idx_pathway + 2))
            skip_connections = getattr(self, "p{}_skip_connections".format(idx_pathway + 2))
            pre_pathway_features = pathway_features[-1]
            cur_pathway_features = []
            for idx, features in enumerate(pre_pathway_features):
                lateral_features = lateral_convs[idx](features)
                sum_count = 0
                if skip_connections[idx]:
                    # skip_features = skip_connections[idx](p1_features[idx])
                    skip_features = p1_features[idx]
                    sum_count += 1
                else:
                    skip_features = 0.0
                if idx < len(pre_pathway_features) - 1 and acrossdown_convs[idx]:
                    acrossdown_features = acrossdown_convs[idx](
                        F.interpolate(pre_pathway_features[idx + 1], scale_factor=2, mode="nearest"))
                        
                    sum_count += 1
                else:
                    acrossdown_features = 0.0
                if idx > 0 and sameup_convs[idx - 1]:
                    sameup_features = sameup_convs[idx - 1](pre_pathway_features[idx - 1])
                    sum_count += 1
                else:
                    sameup_features = 0.0
                # print("lateral: ", lateral_features.shape)
                # if not isinstance(skip_features, float):
                #     print("skip: ", skip_features.shape)
                # if not isinstance(acrossdown_features, float):
                #     print("acrossdown: ", acrossdown_features.shape)
                # if not isinstance(sameup_features, float):
                #     print("sameup: ", sameup_features.shape)
                prev_features = lateral_features + skip_features + acrossdown_features + sameup_features
                
                if self._fuse_type == "avg":
                    prev_features /= sum_count
                cur_pathway_features.append(prev_features)

            pathway_features.append(cur_pathway_features)

        results = []
        for features, output_conv in zip(pathway_features[-1], self.output_convs):
            results.append(output_conv(features))

        # if self.top_block is not None:
        #     top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
        #     if top_block_in_feature is None:
        #         top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
        #     results.extend(self.top_block(top_block_in_feature))
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
def build_resnet_fpg_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPG.IN_FEATURES
    out_channels = cfg.MODEL.FPG.OUT_CHANNELS
    num_pathways = cfg.MODEL.FPG.NUM_PATHWAYS
    backbone = FPG(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_pathways=num_pathways,
        norm=cfg.MODEL.FPG.NORM,
        fuse_type=cfg.MODEL.FPG.FUSE_TYPE,
    )
    return backbone
