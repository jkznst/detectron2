# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_sixdpose_config(cfg):
    """
    Add config for sixdpose head.
    """
    _C = cfg

    _C.MODEL.PVNET_ON = True

    _C.MODEL.ROI_PVNET_HEAD = CN()
    # _C.MODEL.ROI_PVNET_HEAD.NAME = "MaskRCNNConvUpsampleHead"
    _C.MODEL.ROI_PVNET_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
    _C.MODEL.ROI_PVNET_HEAD.NUM_STACKED_CONVS = 8   # densepose is 8
    _C.MODEL.ROI_PVNET_HEAD.CONV_HEAD_DIM = 512     # densepose is 512
    # _C.MODEL.ROI_PVNET_HEAD.CONV_HEAD_KERNEL = 3
    _C.MODEL.ROI_PVNET_HEAD.NORM = ""
    _C.MODEL.ROI_PVNET_HEAD.CLS_AGNOSTIC_MASK = False
    _C.MODEL.ROI_PVNET_HEAD.NUM_KEYPOINTS = 17 # test coco
    
    _C.MODEL.ROI_PVNET_HEAD.DECONV_KERNEL = 4
    _C.MODEL.ROI_PVNET_HEAD.UP_SCALE = 2
    _C.MODEL.ROI_PVNET_HEAD.HEATMAP_SIZE = 56
    _C.MODEL.ROI_PVNET_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_PVNET_HEAD.POOLER_RESOLUTION = 14
    _C.MODEL.ROI_PVNET_HEAD.POOLER_SAMPLING_RATIO = 2
    # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
    _C.MODEL.ROI_PVNET_HEAD.FG_IOU_THRESHOLD = 0.7
    # # Loss weights for annotation masks.(14 Parts)
    _C.MODEL.ROI_PVNET_HEAD.MASK_WEIGHTS = 1.0
    # # Loss weights for VERTEX
    _C.MODEL.ROI_PVNET_HEAD.VERTEX_WEIGHTS = 2.0

    # For Decoder
    _C.MODEL.ROI_PVNET_HEAD.DECODER_ON = False
    _C.MODEL.ROI_PVNET_HEAD.DECODER_NUM_CLASSES = 256
    _C.MODEL.ROI_PVNET_HEAD.DECODER_CONV_DIMS = 256
    _C.MODEL.ROI_PVNET_HEAD.DECODER_NORM = ""
    _C.MODEL.ROI_PVNET_HEAD.DECODER_COMMON_STRIDE = 4

    # ---------------------------------------------------------------------------- #
    # FPG options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.FPG = CN()
    # Names of the input feature maps to be used by FPG
    # They must have contiguous power of 2 strides
    # e.g., ["res2", "res3", "res4", "res5"]
    _C.MODEL.FPG.IN_FEATURES = []
    _C.MODEL.FPG.OUT_CHANNELS = 128 # 128 or 256
    _C.MODEL.FPG.NUM_PATHWAYS = 9

    # Options: "" (no norm), "GN"
    _C.MODEL.FPG.NORM = "BN"

    # Types for fusing the FPG top-down and lateral features. Can be either "sum" or "avg"
    _C.MODEL.FPG.FUSE_TYPE = "sum"

    # test coco
    # _C.INPUT.KEYPOINT_FORMAT = 'bb8+fps8'   # 'bb8', 'fps8', 'bb8+fps8'

    # # `True` if random blur is used for data augmentation during training
    # _C.INPUT.RANDOMBLUR = CN({"ENABLED": True})
    # _C.INPUT.RANDOMBLUR.PROB = 0.5

    # # `True` if color jitter is used for data augmentation during training
    # _C.INPUT.COLORJITTER = CN({"ENABLED": True})
    # _C.INPUT.COLORJITTER.BRIGHTNESS = 0.1
    # _C.INPUT.COLORJITTER.CONTRAST = 0.1
    # _C.INPUT.COLORJITTER.SATURATION = 0.05
    # _C.INPUT.COLORJITTER.HUE = 0.05