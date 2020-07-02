# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_sixdpose_config(cfg):
    """
    Add config for sixdpose head.
    """
    _C = cfg

    _C.MODEL.CRPNET_ON = False
    _C.MODEL.HCR_ON = False
    _C.MODEL.PVNET_ON = False

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
    _C.MODEL.ROI_PVNET_HEAD.POOLER_SAMPLING_RATIO = 0
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

    # comment when train on coco ---------------------------------------------
    _C.INPUT.KEYPOINT_FORMAT = 'bb8+fps8'   # 'bb8', 'fps8', 'bb8+fps8'
    # `True` if random blur is used for data augmentation during training
    _C.INPUT.RANDOMBLUR = CN({"ENABLED": True})
    _C.INPUT.RANDOMBLUR.PROB = 0.5
    # `True` if color jitter is used for data augmentation during training
    _C.INPUT.COLORJITTER = CN({"ENABLED": True})
    _C.INPUT.COLORJITTER.BRIGHTNESS = 0.1
    _C.INPUT.COLORJITTER.CONTRAST = 0.1
    _C.INPUT.COLORJITTER.SATURATION = 0.05
    _C.INPUT.COLORJITTER.HUE = 0.05
    # -------------------------------------------------------------------------

    # ---------------------------------------------------------------------------- #
    # ResNeth options
    # Note that parts of a resnet may be used for both the backbone and the head
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    _C.MODEL.RESNETH = CN()

    _C.MODEL.RESNETH.DEPTH = 50
    _C.MODEL.RESNETH.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone

    # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
    _C.MODEL.RESNETH.NUM_GROUPS = 1
    _C.MODEL.RESNETH.FREEZE_AT = 1

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.RESNETH.NORM = "SyncBN"

    # Baseline width of each group.
    # Scaling this parameters will scale the width of all bottleneck layers.
    # _C.MODEL.RESNETH.WIDTH_PER_GROUP = 64

    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet; use False for C2 and Torch models
    _C.MODEL.RESNETH.STRIDE_IN_1X1 = True

    # Apply dilation in stage "res5"
    _C.MODEL.RESNETH.DILATION_ON_PER_STAGE = [False, False, False, False]

    # Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
    # _C.MODEL.RESNETH.RES2_OUT_CHANNELS = 256
    _C.MODEL.RESNETH.STEM_OUT_CHANNELS = 64

    # Apply Deformable Convolution in stages
    # Specify if apply deform_conv on Res2, Res3, Res4, Res5
    _C.MODEL.RESNETH.DEFORM_ON_PER_STAGE = [False, False, False, False]
    # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
    # Use False for DeformableV1.
    _C.MODEL.RESNETH.DEFORM_MODULATED = False
    # Number of groups in deformable conv.
    _C.MODEL.RESNETH.DEFORM_NUM_GROUPS = 1

    # ---------------------------------------------------------------------------- #
    # CRPNet Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CRPNET = CN()

    _C.MODEL.CRPNET.CASCADE_REGRESSION = True
    _C.MODEL.CRPNET.KPT_WEIGHT = 1.0

    # This is the number of foreground classes.
    _C.MODEL.CRPNET.NUM_CLASSES = 80

    _C.MODEL.CRPNET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

    # Convolutions to use in the cls and bbox tower
    # NOTE: this doesn't include the last conv for logits
    _C.MODEL.CRPNET.NUM_CONVS = 1
    _C.MODEL.CRPNET.NUM_KEYPOINTS = 17 # test coco

    # IoU overlap ratio [bg, fg] for labeling anchors.
    # Anchors with < bg are labeled negative (0)
    # Anchors  with >= bg and < fg are ignored (-1)
    # Anchors with >= fg are labeled positive (1)
    _C.MODEL.CRPNET.IOU_THRESHOLDS = [0.4, 0.5]
    _C.MODEL.CRPNET.IOU_LABELS = [0, -1, 1]

    # Prior prob for rare case (i.e. foreground) at the beginning of training.
    # This is used to set the bias for the logits layer of the classifier subnet.
    # This improves training stability in the case of heavy class imbalance.
    _C.MODEL.CRPNET.PRIOR_PROB = 0.01

    # Inference cls score threshold, only anchors with score > INFERENCE_TH are
    # considered for inference (to improve speed)
    _C.MODEL.CRPNET.SCORE_THRESH_TEST = 0.05
    _C.MODEL.CRPNET.TOPK_CANDIDATES_TEST = 1000
    _C.MODEL.CRPNET.NMS_THRESH_TEST = 0.5

    # Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
    _C.MODEL.CRPNET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    # Loss parameters
    _C.MODEL.CRPNET.FOCAL_LOSS_GAMMA = 2.0
    _C.MODEL.CRPNET.FOCAL_LOSS_ALPHA = 0.25
    _C.MODEL.CRPNET.SMOOTH_L1_LOSS_BETA = 0.1

    # ---------------------------------------------------------------------------- #
    # HCR keypoint Head
    # ---------------------------------------------------------------------------- #
    _C.MODEL.ROI_HCR_HEAD = CN()
    _C.MODEL.ROI_HCR_HEAD.NAME = "HCRConvHead"
    # _C.MODEL.ROI_HCR_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
    _C.MODEL.ROI_HCR_HEAD.NUM_STACKED_CONVS = 4   # densepose is 8
    _C.MODEL.ROI_HCR_HEAD.CONV_HEAD_DIM = 256     # densepose is 512
    # _C.MODEL.ROI_PVNET_HEAD.CONV_HEAD_KERNEL = 3
    _C.MODEL.ROI_HCR_HEAD.NORM = ""
    _C.MODEL.ROI_HCR_HEAD.CLS_AGNOSTIC_KEYPOINT = False
    _C.MODEL.ROI_HCR_HEAD.NUM_KEYPOINTS = 17 # test coco
    _C.MODEL.ROI_HCR_HEAD.KEYPOINT_REG_WEIGHT = 10.0
    
    # _C.MODEL.ROI_HCR_HEAD.DECONV_KERNEL = 4
    # _C.MODEL.ROI_HCR_HEAD.UP_SCALE = 2
    _C.MODEL.ROI_HCR_HEAD.HEATMAP_SIZE = 56
    _C.MODEL.ROI_HCR_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_HCR_HEAD.POOLER_RESOLUTION = 14
    _C.MODEL.ROI_HCR_HEAD.POOLER_SAMPLING_RATIO = 0
    # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
    _C.MODEL.ROI_HCR_HEAD.FG_IOU_THRESHOLD = 0.7
    # # Loss weights for annotation masks.(14 Parts)
    _C.MODEL.ROI_HCR_HEAD.HEATMAP_WEIGHTS = 1.0
    # # Loss weights for VERTEX
    _C.MODEL.ROI_HCR_HEAD.OFFSET_WEIGHTS = 2.0

    # For Transition module
    _C.MODEL.ROI_HCR_HEAD.TRANSITION_ON = False
    _C.MODEL.ROI_HCR_HEAD.TRANSITION_NUM_CLASSES = 256
    _C.MODEL.ROI_HCR_HEAD.TRANSITION_CONV_DIMS = 256
    _C.MODEL.ROI_HCR_HEAD.TRANSITION_NORM = ""
    _C.MODEL.ROI_HCR_HEAD.TRANSITION_COMMON_STRIDE = 4