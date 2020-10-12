# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .config import add_sixdpose_config
from .pvnet_head import ROI_PVNET_HEAD_REGISTRY
from .hcr_head import ROI_HCR_HEAD_REGISTRY
from .roi_head import SixDPoseROIHeads, HCRROIHeads

from .dataset_mapper import DatasetMapper, COCODatasetMapper
# comment when train on coco
from . import dataset  # just to register data
from .coco_evaluator import COCOEvaluator
from .pose_evaluator import SixDPoseLinemodEvaluator

from .fpg import build_resnet_fpg_backbone, FPG
from .resneth import build_crpnet_resneth_fpn_backbone, build_crpnet_resnet_fpn_backbone
from .crpnet import CRPNet