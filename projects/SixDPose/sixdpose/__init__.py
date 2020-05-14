# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .config import add_sixdpose_config
from .pvnet_head import ROI_PVNET_HEAD_REGISTRY
from .roi_head import SixDPoseROIHeads

# test coco
# from . import dataset  # just to register data
from .dataset_mapper import DatasetMapper, COCODatasetMapper
# from .coco_evaluator import COCOEvaluator
# from .pose_evaluator import SixDPoseEvaluator

from .fpg import build_resnet_fpg_backbone, FPG
from .resneth import build_crpnet_resneth_fpn_backbone, build_crpnet_resnet_fpn_backbone
from .crpnet import CRPNet