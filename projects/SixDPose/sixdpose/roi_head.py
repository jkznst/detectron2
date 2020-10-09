# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads, ROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals

from .pvnet_head import (
    build_pvnet_data_filter,
    build_pvnet_head,
    build_pvnet_losses,
    pvnet_inference,
)

from .hcr_head import (
    build_hcr_data_filter,
    build_hcr_head,
    build_hcr_losses,
    hcr_inference,
)

from .resneth import BottleneckBlock

class Decoder(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features):
        super(Decoder, self).__init__()

        # fmt: off
        self.in_features      = in_features
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        num_classes           = cfg.MODEL.ROI_PVNET_HEAD.DECODER_NUM_CLASSES
        conv_dims             = cfg.MODEL.ROI_PVNET_HEAD.DECODER_CONV_DIMS
        self.common_stride    = cfg.MODEL.ROI_PVNET_HEAD.DECODER_COMMON_STRIDE
        norm                  = cfg.MODEL.ROI_PVNET_HEAD.DECODER_NORM
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features):
        for i, _ in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        x = self.predictor(x)
        return x

class FeatureTransitionModule(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features):
        super(FeatureTransitionModule, self).__init__()

        # fmt: off
        self.in_features      = in_features
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        # self.common_stride    = cfg.MODEL.ROI_HCR_HEAD.DECODER_COMMON_STRIDE
        norm                  = cfg.MODEL.ROI_HCR_HEAD.TRANSITION_NORM
        # fmt: on

        self.transitions = []
        for in_feature in self.in_features:
            trans_ops = []
            trans_length = max(
                1, int(np.log2(feature_strides[in_feature]) - 1)
            )
            for k in range(trans_length):
                trans_ops.append(BottleneckBlock(
                    in_channels=feature_channels[in_feature],
                    out_channels=feature_channels[in_feature],
                    bottleneck_channels=feature_channels[in_feature] // 4,
                    stride=1,
                    num_groups=1,
                    stride_in_1x1=False,
                    dilation=1,
                ))
            self.transitions.append(nn.Sequential(*trans_ops))
            self.add_module(in_feature, self.transitions[-1])

        self.lateral = []
        self.down_path = []
        self.up_path = []
        for i in range(len(in_features) - 1):
            lateral_conv = Conv2d(
                    feature_channels[in_features[i + 1]],
                    feature_channels[in_features[i + 1]],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=not norm,
                    norm=get_norm(norm, feature_channels[in_features[i + 1]]),
                    activation=F.relu,
                )
            down_conv = Conv2d(
                    feature_channels[in_features[i + 1]],
                    feature_channels[in_features[i]],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, feature_channels[in_features[i]]),
                    activation=F.relu,
                )
            up_conv = Conv2d(
                    feature_channels[in_features[i]],
                    feature_channels[in_features[i + 1]],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, feature_channels[in_features[i + 1]]),
                    activation=F.relu,
                )
            weight_init.c2_msra_fill(lateral_conv)
            weight_init.c2_msra_fill(down_conv)
            weight_init.c2_msra_fill(up_conv)
            self.lateral.append(lateral_conv)
            self.down_path.append(down_conv)
            self.up_path.append(up_conv)
            self.add_module("lateral{}".format(i), lateral_conv)
            self.add_module("down_path{}".format(i), down_conv)
            self.add_module("up_path{}".format(i), up_conv)

    def forward(self, features):
        trans_features = []
        for i, _ in enumerate(self.in_features):
            x = self.transitions[i](features[i])
            trans_features.append(x)
        fused_features = []
        for i, _ in enumerate(self.in_features):
            if i == 0:
                lateral_feat = trans_features[i]
                down_feat = self.down_path[i](
                    F.interpolate(trans_features[i + 1], scale_factor=2, mode="nearest"))
                fused_feat = lateral_feat + down_feat + features[i]
                fused_features.append(fused_feat)
            elif i == len(self.in_features) - 1:
                lateral_feat = self.lateral[i - 1](trans_features[i])
                up_feat = self.up_path[i - 1](fused_features[i - 1])
                fused_feat = lateral_feat + up_feat + features[i]
                fused_features.append(fused_feat)
            else:
                lateral_feat = self.lateral[i - 1](trans_features[i])
                down_feat = self.down_path[i](
                    F.interpolate(trans_features[i + 1], scale_factor=2, mode="nearest"))
                up_feat = self.up_path[i - 1](fused_features[i - 1])
                fused_feat = lateral_feat + down_feat + up_feat + features[i]
                fused_features.append(fused_feat)
        return fused_features
        

@ROI_HEADS_REGISTRY.register()
class SixDPoseROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of PVNet head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_pvnet_head(cfg, input_shape)

    def _init_pvnet_head(self, cfg, input_shape):
        # fmt: off
        self.pvnet_on          = cfg.MODEL.PVNET_ON
        if not self.pvnet_on:
            return
        self.pvnet_data_filter = build_pvnet_data_filter(cfg)
        pvnet_pooler_resolution       = cfg.MODEL.ROI_PVNET_HEAD.POOLER_RESOLUTION
        # pvnet_pooler_scales           = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        pvnet_pooler_sampling_ratio   = cfg.MODEL.ROI_PVNET_HEAD.POOLER_SAMPLING_RATIO
        pvnet_pooler_type             = cfg.MODEL.ROI_PVNET_HEAD.POOLER_TYPE
        self.use_decoder              = cfg.MODEL.ROI_PVNET_HEAD.DECODER_ON
        # fmt: on
        if self.use_decoder:
            pvnet_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        else:
            pvnet_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = Decoder(cfg, input_shape, self.in_features)

        self.pvnet_pooler = ROIPooler(
            output_size=pvnet_pooler_resolution,
            scales=pvnet_pooler_scales,
            sampling_ratio=pvnet_pooler_sampling_ratio,
            pooler_type=pvnet_pooler_type,
        )
        self.pvnet_head = build_pvnet_head(cfg, in_channels)
        # self.pvnet_predictor = build_pvnet_predictor(
        #     cfg, self.pvnet_head.n_out_channels
        # )
        self.pvnet_losses = build_pvnet_losses(cfg)

    def _forward_pvnet(self, features, instances):
        """
        Forward logic of the pvnet prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.pvnet_on:
            return {} if self.training else instances

        # features = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals_pvnet = self.pvnet_data_filter(proposals)
            # proposals_pvnet = proposals
            # print(len(proposals_pvnet[0]))
            if len(proposals_pvnet) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals_pvnet]

                if self.use_decoder:
                    features = [self.decoder(features)]

                features_pvnet = self.pvnet_pooler(features, proposal_boxes)
                pvnet_head_outputs = self.pvnet_head(features_pvnet)
                # pvnet_outputs, _ = self.pvnet_predictor(pvnet_head_outputs)
                pvnet_loss_dict = self.pvnet_losses(proposals_pvnet, pvnet_head_outputs)
                return pvnet_loss_dict
            elif len(proposals_pvnet) == 0:
                return {}
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            if self.use_decoder:
                features = [self.decoder(features)]
                
            features_pvnet = self.pvnet_pooler(features, pred_boxes)
            if len(features_pvnet) > 0:
                pvnet_outputs = self.pvnet_head(features_pvnet)
                # pvnet_outputs, _ = self.pvnet_predictor(pvnet_head_outputs)
            else:
                # If no detection occurred instances
                # set pvnet_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_pvnet.device)
                pvnet_outputs = {'mask': empty_tensor, 'vertex': empty_tensor}

            pvnet_inference(pvnet_outputs, instances)
            return instances

    def forward(self, images, features, proposals, targets=None):
        features_list = [features[f] for f in self.in_features]

        instances, losses = super().forward(images, features, proposals, targets)
        del targets, images

        if self.training:
            losses.update(self._forward_pvnet(features_list, instances))
        else:
            instances = self._forward_pvnet(features_list, instances)
        return instances, losses


@ROI_HEADS_REGISTRY.register()
class HCRROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of HCR head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_hcr_head(cfg, input_shape)

    def _init_hcr_head(self, cfg, input_shape):
        # fmt: off
        self.hcr_on          = cfg.MODEL.HCR_ON
        if not self.hcr_on:
            return
        self.hcr_data_filter = build_hcr_data_filter(cfg)
        hcr_pooler_resolution       = cfg.MODEL.ROI_HCR_HEAD.POOLER_RESOLUTION
        hcr_pooler_sampling_ratio   = cfg.MODEL.ROI_HCR_HEAD.POOLER_SAMPLING_RATIO
        hcr_pooler_type             = cfg.MODEL.ROI_HCR_HEAD.POOLER_TYPE
        self.use_decoder              = cfg.MODEL.ROI_HCR_HEAD.TRANSITION_ON
        # fmt: on
        # if self.use_decoder:
        #     hcr_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        # else:
        #     hcr_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        hcr_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = FeatureTransitionModule(cfg, input_shape, self.in_features)

        self.hcr_pooler = ROIPooler(
            output_size=hcr_pooler_resolution,
            scales=hcr_pooler_scales,
            sampling_ratio=hcr_pooler_sampling_ratio,
            pooler_type=hcr_pooler_type,
        )
        self.hcr_head = build_hcr_head(cfg, in_channels)
        # self.pvnet_predictor = build_pvnet_predictor(
        #     cfg, self.pvnet_head.n_out_channels
        # )
        self.hcr_losses = build_hcr_losses(cfg)

    def _forward_hcr(self, features, instances):
        """
        Forward logic of the pvnet prediction branch.

        Args:
            features (list[Tensor]): #level input features for densepose prediction
            instances (list[Instances]): the per-image instances to train/predict densepose.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.hcr_on:
            return {} if self.training else instances

        # features = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            # proposals_hcr = self.hcr_data_filter(proposals)
            proposals_hcr = proposals
            # print(len(proposals_pvnet[0]))
            if len(proposals_hcr) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals_hcr]

                if self.use_decoder:
                    features = self.decoder(features)

                features_hcr = self.hcr_pooler(features, proposal_boxes)
                hcr_head_outputs = self.hcr_head(features_hcr)
                # pvnet_outputs, _ = self.pvnet_predictor(pvnet_head_outputs)
                hcr_loss_dict = self.hcr_losses(proposals_hcr, hcr_head_outputs)
                return hcr_loss_dict
            elif len(proposals_hcr) == 0:
                return {}
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            if self.use_decoder:
                features = self.decoder(features)
                
            features_hcr = self.hcr_pooler(features, pred_boxes)
            if len(features_hcr) > 0:
                hcr_outputs = self.hcr_head(features_hcr)
                # pvnet_outputs, _ = self.pvnet_predictor(pvnet_head_outputs)
            else:
                # If no detection occurred instances
                # set pvnet_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_hcr.device)
                hcr_outputs = {'heatmap': empty_tensor, 'offset': empty_tensor, 'variance': empty_tensor}

            hcr_inference(hcr_outputs, instances)
            return instances, features

    def forward(self, images, features, proposals, targets=None):
        features_list = [features[f] for f in self.in_features]

        instances, losses = super().forward(images, features, proposals, targets)
        del targets, images

        if self.training:
            losses.update(self._forward_hcr(features_list, instances))
        else:
            instances, features = self._forward_hcr(features_list, instances)
            return instances, losses, features
        return instances, losses