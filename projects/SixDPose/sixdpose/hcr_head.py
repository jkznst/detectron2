# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, interpolate, get_norm
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.structures import heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

# from .csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer_v3


_TOTAL_SKIPPED = 0

ROI_HCR_HEAD_REGISTRY = Registry("ROI_HCR_HEAD")
ROI_HCR_HEAD_REGISTRY.__doc__ = """
Registry for HCR keypoint heads, which make keypoint predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_hcr_head(cfg, input_channels):
    head_name = cfg.MODEL.ROI_HCR_HEAD.NAME
    return ROI_HCR_HEAD_REGISTRY.get(head_name)(cfg, input_channels)


@ROI_HCR_HEAD_REGISTRY.register()
class HCRConvHead(nn.Module):
    """
    A HCR keypoint head with several conv layers.
    """

    def __init__(self, cfg, input_channels):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(HCRConvHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        num_keypoints     = cfg.MODEL.ROI_HCR_HEAD.NUM_KEYPOINTS
        conv_dims         = cfg.MODEL.ROI_HCR_HEAD.CONV_HEAD_DIM
        self.norm         = cfg.MODEL.ROI_HCR_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_HCR_HEAD.NUM_STACKED_CONVS
        # input_channels    = input_shape.channels
        cls_agnostic_kpt = cfg.MODEL.ROI_HCR_HEAD.CLS_AGNOSTIC_KEYPOINT
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            if k < num_conv - 1:
                conv = Conv2d(
                    input_channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dims),
                    activation=F.relu,
                )
            else:
                conv = Conv2d(
                    input_channels if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=not self.norm,
                    norm=get_norm(self.norm, conv_dims),
                    activation=F.relu,
                )
            self.add_module("hcr_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        # self.deconv = ConvTranspose2d(
        #     conv_dims if num_conv > 0 else input_channels,
        #     conv_dims,
        #     kernel_size=2,
        #     stride=2,
        #     padding=0,
        # )

        num_heatmaps = num_keypoints if cls_agnostic_kpt else num_classes * num_keypoints
        num_offsets = 2 * num_keypoints if cls_agnostic_kpt else 2 * num_classes * num_keypoints
        
        self.heatmap_predictor = Conv2d(conv_dims, num_heatmaps, kernel_size=1, stride=1, padding=0)
        self.offset_predictor = Conv2d(conv_dims, num_offsets, kernel_size=1, stride=1, padding=0)
        self.variance_predictor = Conv2d(conv_dims, num_offsets, kernel_size=1, stride=1, padding=0)

        # for layer in self.conv_norm_relus + [self.deconv]:
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for heatmap prediction layer
        nn.init.normal_(self.heatmap_predictor.weight, std=0.001)
        if self.heatmap_predictor.bias is not None:
            nn.init.constant_(self.heatmap_predictor.bias, 0)
        # use normal distribution initialization for offset prediction layer
        nn.init.normal_(self.offset_predictor.weight, std=0.001)
        if self.offset_predictor.bias is not None:
            nn.init.constant_(self.offset_predictor.bias, 0)
        # use normal distribution initialization for variance prediction layer
        nn.init.normal_(self.variance_predictor.weight, std=0.001)
        if self.variance_predictor.bias is not None:
            nn.init.constant_(self.variance_predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        # x = F.relu(self.deconv(x))
        return {"heatmap": self.heatmap_predictor(x),
                "offset": self.offset_predictor(x),
                "variance": self.variance_predictor(x)}


class HCRDataFilter(object):
    def __init__(self, cfg):
        self.iou_threshold = cfg.MODEL.ROI_HCR_HEAD.FG_IOU_THRESHOLD

    @torch.no_grad()
    def __call__(self, proposals_with_targets):
        """
        Filters proposals with targets to keep only the ones relevant for
        Pose training
        proposals: list(Instances), each element of the list corresponds to
            various instances (proposals, GT for boxes and densepose) for one
            image
        """
        proposals_filtered = []
        for proposals_per_image in proposals_with_targets:
            # print(proposals_per_image)
            if not hasattr(proposals_per_image, "gt_keypoints"):
                continue
            assert hasattr(proposals_per_image, "gt_boxes")
            assert hasattr(proposals_per_image, "proposal_boxes")
            gt_boxes = proposals_per_image.gt_boxes
            est_boxes = proposals_per_image.proposal_boxes
            # apply match threshold for densepose head
            iou = matched_boxlist_iou(gt_boxes, est_boxes)
            iou_select = iou > self.iou_threshold
            proposals_per_image = proposals_per_image[iou_select]
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            # filter out any target without densepose annotation
            # gt_keypoints = proposals_per_image.gt_keypoints
            # assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.gt_keypoints)
            # selected_indices = [
            #     i for i, kpt_target in enumerate(gt_keypoints) if kpt_target is not None
            # ]
            # if len(selected_indices) != len(gt_keypoints):
            #     proposals_per_image = proposals_per_image[selected_indices]
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.proposal_boxes)
            assert len(proposals_per_image.gt_boxes) == len(proposals_per_image.gt_keypoints)
            # print(proposals_per_image)
            proposals_filtered.append(proposals_per_image)
        return proposals_filtered

def build_hcr_data_filter(cfg):
    hcr_filter = HCRDataFilter(cfg)
    return hcr_filter

@torch.no_grad()
def decode_keypoints(offsets, heatmap_probs, variances, rois, kpt_reg_weight=10.0):
    '''
    Arguments:
        offsets: shape (B, 2*num_kpt, H, W)
        heatmap_probs: shape (B, num_kpt, H, W)
        variances: shape (B, 2*num_kpt, H, W)
        rois: shape (B, 4), xyxy
    return:
        keypoints: shape (B, num_pkt, 3)
    '''
    assert offsets.size(0) == rois.size(0), "batch size dont match"
    if rois.numel() == 0:
        return rois.new().long()

    n = heatmap_probs.size(0)
    num_kpt = heatmap_probs.size(1)
    heatmap_size = heatmap_probs.size(2)

    lt_x = rois[:, 0]
    lt_y = rois[:, 1]
    rois_width = rois[:, 2] - rois[:, 0]
    rois_height = rois[:, 3] - rois[:, 1]
    interval_x = rois_width / (heatmap_size - 1)
    interval_y = rois_height / (heatmap_size - 1)

    # grid_x = torch.arange(heatmap_size).repeat(heatmap_size, 1).reshape(1, heatmap_size, heatmap_size).to(device=rois.device)
    # grid_x = grid_x * interval_x.reshape(n, 1, 1) + lt_x.reshape(n, 1, 1)
    # grid_y = torch.arange(heatmap_size).repeat(heatmap_size, 1).transpose(0, 1).reshape(1, heatmap_size, heatmap_size).to(device=rois.device)
    # grid_y = grid_y * interval_y.reshape(n, 1, 1) + lt_y.reshape(n, 1, 1)
    # grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(n, 1, heatmap_size, heatmap_size, 2)

    heatmap_probs_ = heatmap_probs.reshape(n * num_kpt, -1)
    max_probs, indices_probs = torch.max(heatmap_probs_, dim=-1, keepdim=False)
    indices_probs_x = indices_probs % heatmap_size
    indices_probs_y = indices_probs // heatmap_size
    grid_x = indices_probs_x.reshape(n, num_kpt) * interval_x[:, None] + lt_x[:, None]
    grid_y = indices_probs_y.reshape(n, num_kpt) * interval_y[:, None] + lt_y[:, None]
    grid_xy = torch.stack([grid_x, grid_y], dim=-1) # shape (n, num_kpt, 2)

    offsets_x = offsets[:, 0::2].reshape(n * num_kpt, -1)
    offsets_y = offsets[:, 1::2].reshape(n * num_kpt, -1)
    tmp_indices = torch.arange(offsets_x.size(0), device=offsets.device)
    offsets_x = offsets_x[tmp_indices, indices_probs].reshape(n, num_kpt)
    offsets_y = offsets_y[tmp_indices, indices_probs].reshape(n, num_kpt)
    offsets_x = offsets_x / kpt_reg_weight / heatmap_size * rois_width[:, None]
    offsets_y = offsets_y / kpt_reg_weight / heatmap_size * rois_height[:, None]
    offsets_xy = torch.stack([offsets_x, offsets_y], dim=-1)    # shape (n, num_kpt, 2)

    variances_x = variances[:, 0::2].reshape(n * num_kpt, -1)
    variances_y = variances[:, 1::2].reshape(n * num_kpt, -1)
    variances_x = variances_x[tmp_indices, indices_probs].reshape(n, num_kpt)
    variances_y = variances_y[tmp_indices, indices_probs].reshape(n, num_kpt)
    variances_xy = torch.stack([variances_x, variances_y], dim=-1) # shape (n, num_kpt, 2)

    kpt_2d = grid_xy + offsets_xy
    kpt_2d = torch.cat((kpt_2d, variances_xy), dim=-1)   # shape (n, num_kpt, 4)
    # print(kpt_2d[0])
    return kpt_2d

def hcr_inference(hcr_outputs, pred_instances):
    """
    Convert pred_heatmap to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        hcr_outputs: {pred_heatmap, pred_offset, pred_var}
                pred_heatmap (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
                pred_offset, pred_var (Tensor):
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    pred_heatmap, pred_offset, pred_var = hcr_outputs["heatmap"], hcr_outputs["offset"], hcr_outputs["variance"] 
    
    if pred_heatmap.size(0) > 0:
        cls_agnostic_kpt = True #pred_heatmap.size(1) == 1
        if cls_agnostic_kpt:
            heatmap_probs_pred = pred_heatmap.sigmoid()
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_heatmap.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            indices = torch.arange(num_masks, device=class_pred.device)
            heatmap_probs_pred = pred_heatmap[indices, class_pred][:, None].sigmoid()
        # heatmap_probs_pred.shape: (B, 1, Hmask, Wmask)

        num_boxes_per_image = [len(i) for i in pred_instances]
        # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
        bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)
        keypoint_results = decode_keypoints(pred_offset.detach(), heatmap_probs_pred.detach(),
                            pred_var.detach(), bboxes_flat.detach())

        # keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_boxes_per_image, dim=0)
        keypoint_results = keypoint_results.split(num_boxes_per_image, dim=0)

        for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
            # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y)
            instances_per_image.pred_keypoints = keypoint_results_per_image
    # elif pred_heatmap.size(0) == 0:
    #     for instances in pred_instances:
    #         instances.pred_masks = torch.zeros(size=(0, 0, 0, 0), device=pred_heatmap.device)
    #         instances.pred_keypoints = torch.zeros(size=(0, 0, 0), device=pred_offset.device)

def _neg_loss(pred, gt, mask=None, HHKM=False):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
      mask (batch x c x h x w)
      HHKM: bool, hurestic hard keypoint mining
  '''
  if mask is not None:
    pos_inds = gt.eq(1).float() * mask
    neg_inds = gt.lt(1).float() * mask
  else:
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0
#   print(pred[0, 0])
#   print(gt[0, 0])

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  if HHKM:
    km_weights = gt.float().sum(dim=[2, 3])
    km_weights = 1.0 - 1.0 / km_weights 
    km_normalizer = km_weights.mean()
    pos_loss = pos_loss.sum(dim=[2, 3]) * km_weights / km_normalizer
    neg_loss = neg_loss.sum(dim=[2, 3]) * km_weights / km_normalizer

  num_pos  = pos_inds.float().sum()
#   print(num_pos)
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target, mask=None):
    if mask is not None:
      mask = mask.expand_as(out).float()
    return self.neg_loss(out, target, mask)

class DenseRegL1Loss(nn.Module):
    def __init__(self, norm=False, smooth=False, mask_thresh=0.9, mask_power=4):
        super(DenseRegL1Loss, self).__init__()
        self.norm = norm
        self.mask_thresh = mask_thresh
        self.mask_power = mask_power
        if smooth:
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask):
        if self.norm:
            pred = pred / (target + 1e-4)
            target = target * 0 + 1
        # ignore and degrade faraway points
        mask = (mask.ge(self.mask_thresh).float() * mask.pow(self.mask_power)).expand_as(pred)
        # print(mask.sum())
        loss = self.loss_fn(pred, target) * mask
        loss = loss.sum() / (mask.sum() + 1e-4)
        return loss

class DenseKLLoss(nn.Module):
    def __init__(self, norm=False, mask_thresh=0.9, mask_power=4, ohcm_topk=-1):
        super(DenseKLLoss, self).__init__()
        self.norm = norm
        self.mask_thresh = mask_thresh
        self.mask_power = mask_power
        self.ohcm_topk = ohcm_topk

    def forward(self, pred, target, variance, mask):
        if self.norm:
            pred = pred / (target + 1e-4)
            target = target * 0 + 1

        abs_error = torch.abs(pred - target)

        greater_loss = torch.exp(-variance) * (abs_error - 0.5) + 0.5 * variance
        lower_loss = 0.5 * torch.exp(-variance) * (abs_error * abs_error) + 0.5 * variance

        condition = abs_error.ge(1.0)
        kl_loss = torch.where(condition, greater_loss, lower_loss)
        # ignore and degrade faraway points
        mask = (mask.ge(self.mask_thresh).float() * mask.pow(self.mask_power)).expand_as(pred)
        # print(mask[0, 0])
        # print(mask.sum())
        
        if self.ohcm_topk > 0:
            abs_error_ = torch.sum(abs_error * mask, dim=(2, 3))
            sorted_error, indices_error = torch.topk(abs_error_, self.ohcm_topk, dim=-1, 
                                largest=True, sorted=True)
            ohcm_mask = torch.zeros_like(abs_error_)
            tmp_indices = torch.arange(abs_error_.size(0), device=abs_error_.device)[:, None]
            ohcm_mask[tmp_indices, indices_error] = 1.0

            mask = mask * ohcm_mask[..., None, None]
        
        kl_loss = kl_loss * mask
        kl_loss = kl_loss.sum() / (mask.sum() + 1e-4)
        
        return kl_loss

@torch.no_grad()
def keypoints_to_hm_offset(
    keypoints: torch.Tensor, rois: torch.Tensor, heatmap_size: int, kpt_reg_weight=10.0):
    """
    Encode keypoint locations into a hybrid of heatmap and offset for use in hcr kpt head.

    Arguments:
        keypoints: tensor of keypoint locations in of shape (N, K, 3). float
        rois: Nx4 tensor of rois in xyxy format
        heatmap_size: integer side length of square heatmap.

    Returns:
        heatmaps: A tensor of shape (N, K, H, W) encoding keypoint coordinates.
        offsets: A tensor of shape (N, K * 2, H, W) encoding keypoint coordinates.
        valid: A tensor of shape (N, K * 2) containing whether each keypoint is valid or not.
    """
    assert keypoints.size(0) == rois.size(0), "sizes dont match"
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long(), rois.new().long()

    n = keypoints.size(0)
    num_kpt = keypoints.size(1)
    hm_topk = 4

    lt_x = rois[:, 0]
    lt_y = rois[:, 1]
    rois_width = rois[:, 2] - rois[:, 0]
    rois_height = rois[:, 3] - rois[:, 1]
    interval_x = rois_width / (heatmap_size - 1)
    interval_y = rois_height / (heatmap_size - 1)

    grid_x = torch.arange(heatmap_size).repeat(heatmap_size, 1).reshape(1, heatmap_size, heatmap_size).to(device=rois.device)
    grid_x = grid_x * interval_x.reshape(n, 1, 1) + lt_x.reshape(n, 1, 1)
    grid_y = torch.arange(heatmap_size).repeat(heatmap_size, 1).transpose(0, 1).reshape(1, heatmap_size, heatmap_size).to(device=rois.device)
    grid_y = grid_y * interval_y.reshape(n, 1, 1) + lt_y.reshape(n, 1, 1)
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(n, 1, heatmap_size, heatmap_size, 2)

    # kpt_x = keypoints[..., 0]
    # kpt_y = keypoints[..., 1]
    # kpt_x = (kpt_x - lt_x[:, None]) / interval_x
    # kpt_y = (kpt_y - lt_y[:, None]) / interval_y
    kpt_xy = keypoints[..., 0:2].reshape(n, num_kpt, 1, 1, 2).to(device=rois.device)

    offset_xy = kpt_xy - grid_xy    # (N, K, H, H, 2)
    offset_xy = offset_xy / torch.stack([rois_width, rois_height], dim=-1).reshape(n, 1, 1, 1, 2) * heatmap_size
    offset_xy = offset_xy * kpt_reg_weight
    distance = torch.norm(offset_xy, dim=-1, keepdim=False).reshape(n * num_kpt, -1)  # (N*K, H*H)
    heatmap = torch.zeros_like(distance)
    sorted_dist, indices_dist = torch.topk(distance, hm_topk, dim=-1, 
                                largest=False, sorted=True)
    
    tmp_indices = torch.arange(distance.size(0), device=rois.device)[:, None]
    # strategy 1
    # heatmap[tmp_indices, indices_dist] = 1.0
    # strategy 2
    p_dist = F.softmin(sorted_dist, dim=-1)
    p_normalizer, _ = torch.max(p_dist, dim=-1, keepdim=True)
    p_dist = p_dist / p_normalizer
    heatmap[tmp_indices, indices_dist] = p_dist

    # print(distance[0].reshape(heatmap_size, heatmap_size))
    # print(sorted_dist[0])
    # print(indices_dist[0])
    # print(heatmap[0].reshape(heatmap_size, heatmap_size))
    
    # verify
    # print(keypoints)
    # print(torch.stack([x, y], dim=-1))
    # vertexs = vertexs.permute(0, 2, 3, 1)
    # b, h, w, vn_2 = vertexs.shape
    # vertexs = vertexs.view(b, h, w, vn_2//2, 2)
    # # masks = (mask_probs > mask_prob_threshold).squeeze(dim=1)
    # kpt_2d = ransac_voting_layer_v3(masks, vertexs, 128, inlier_thresh=0.99, max_num=100)
    # print(kpt_2d)
    # kpt_2d[..., 0] = kpt_2d[..., 0] / scale_x + offset_x
    # kpt_2d[..., 1] = kpt_2d[..., 1] / scale_y + offset_y
    # print(kpt_2d)

    vis = keypoints[..., 2] > 0
    valid = vis.long()
    # shape (N, 2*K)
    valid = valid.repeat(1, 2).reshape(valid.size(0), 2, -1).permute(0, 2, 1).reshape(valid.size(0), -1)

    # shape (N, K, H, H, 2) -> (N, 2*K, H, H)
    offset_xy = offset_xy.permute(0, 2, 3, 1, 4).reshape(n, heatmap_size, heatmap_size, -1).permute(0, 3, 1, 2)
    # shape (N*K, H*H) -> (N, K, H, H)
    heatmap = heatmap.reshape(n, num_kpt, heatmap_size, heatmap_size)

    # print(valid[0])
    # print(offset_xy[0, 0])
    # print(heatmap[0, 0])
    return heatmap, offset_xy, valid

class HCRLosses(object):
    def __init__(self, cfg):
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_HCR_HEAD.HEATMAP_SIZE
        self.w_heatmap    = cfg.MODEL.ROI_HCR_HEAD.HEATMAP_WEIGHTS
        self.w_offset     = cfg.MODEL.ROI_HCR_HEAD.OFFSET_WEIGHTS
        self.num_classes  = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.num_keypoints = cfg.MODEL.ROI_HCR_HEAD.NUM_KEYPOINTS
        self.kpt_reg_weight= cfg.MODEL.ROI_HCR_HEAD.KEYPOINT_REG_WEIGHT
        # fmt: on

    def __call__(self, proposals_with_gt, hcr_outputs, vis_period=0):
        """
        Compute the hcr keypoint prediction loss.

        Args:
            hcr_outputs: {pred_heatmap, pred_offset, pred_var}
                pred_heatmap (Tensor): A tensor of shape (B, K*C, Hmask, Wmask) or (B, K, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
                pred_offset (Tensor):
            proposals_with_gt (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1 correspondence with the pred_heatmap. 
                The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.

        Returns:
            kpt_loss (Tensor): A scalar tensor containing the loss.
        """
        losses = {}
        
        pred_heatmap, pred_offset, pred_var = hcr_outputs["heatmap"], hcr_outputs["offset"], hcr_outputs["variance"]
        
        cls_agnostic_kpt = pred_heatmap.size(1) == self.num_keypoints
        total_num_instances = pred_heatmap.size(0)
        assert self.heatmap_size == pred_heatmap.size(2)
        heatmap_side_len = self.heatmap_size
        assert pred_heatmap.size(2) == pred_heatmap.size(3), "Heatmap prediction must be square!"

        gt_classes = []
        gt_heatmaps = []
        gt_offsets = []
        gt_valid = []
        for instances_per_image in proposals_with_gt:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_kpt:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            # compute heatmap and offset targets
            gt_hm_per_image, gt_offsets_per_image, gt_valid_per_image = keypoints_to_hm_offset(
                instances_per_image.gt_keypoints.tensor,
                instances_per_image.proposal_boxes.tensor, heatmap_side_len,
                self.kpt_reg_weight
            )
            # verify keypoint encoding and decoding calculations
            # print(instances_per_image.gt_keypoints.tensor[0])
            # kpt_ = decode_keypoints(offsets=gt_offsets_per_image, 
            # heatmap_probs=gt_hm_per_image, 
            # variances=gt_offsets_per_image, rois=instances_per_image.proposal_boxes.tensor)
            # print(kpt_[0])

            gt_hm_per_image = gt_hm_per_image.to(device=pred_heatmap.device)
            gt_offsets_per_image = gt_offsets_per_image.to(device=pred_offset.device)
            gt_valid_per_image = gt_valid_per_image.to(device=pred_offset.device)
            gt_heatmaps.append(gt_hm_per_image)
            gt_offsets.append(gt_offsets_per_image)
            gt_valid.append(gt_valid_per_image)

        if len(gt_heatmaps) == 0:
            return pred_heatmap.sum() * 0

        gt_heatmaps = cat(gt_heatmaps, dim=0) # shape (N, K, H, W)
        gt_offsets = cat(gt_offsets, dim=0) # shape (N, 2K, H, W)
        gt_valid = cat(gt_valid, dim=0) # shape (N, 2K)

        if cls_agnostic_kpt:
            pred_heatmap = pred_heatmap[:, 0:self.num_keypoints]
        else:
            indices = torch.arange(total_num_instances)
            gt_classes = cat(gt_classes, dim=0)
            pred_heatmap = pred_heatmap[indices, gt_classes]

        if gt_heatmaps.dtype == torch.bool:
            gt_heatmaps_bool = gt_heatmaps
        else:
            # Here we allow gt_heatmaps to be float as well (depend on the implementation of rasterize())
            gt_heatmaps_bool = gt_heatmaps > 0.5
        gt_heatmaps = gt_heatmaps.to(dtype=torch.float32)
        gt_offsets = gt_offsets.to(dtype=torch.float32)
        gt_valid = gt_valid.to(dtype=torch.float32)

        # Log the training accuracy (using gt classes and 0.5 threshold)
        # mask_incorrect = (pred_heatmap > 0.0) != gt_heatmaps_bool
        # mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        # num_positive = gt_heatmaps_bool.sum().item()
        # false_positive = (mask_incorrect & ~gt_heatmaps_bool).sum().item() / max(
        #     gt_heatmaps_bool.numel() - num_positive, 1.0
        # )
        # false_negative = (mask_incorrect & gt_heatmaps_bool).sum().item() / max(num_positive, 1.0)

        # storage = get_event_storage()
        # storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
        # storage.put_scalar("mask_rcnn/false_positive", false_positive)
        # storage.put_scalar("mask_rcnn/false_negative", false_negative)
        # if vis_period > 0 and storage.iter % vis_period == 0:
        #     pred_masks = pred_heatmap.sigmoid()
        #     vis_masks = torch.cat([pred_masks, gt_heatmaps], axis=2)
        #     name = "Left: mask prediction;   Right: mask GT"
        #     for idx, vis_mask in enumerate(vis_masks):
        #         vis_mask = torch.stack([vis_mask] * 3, axis=0)
        #         storage.put_image(name + f" ({idx})", vis_mask)

        self.hm_crit = FocalLoss()
        # heatmap_loss = F.binary_cross_entropy_with_logits(pred_heatmap, gt_heatmaps, reduction="mean")
        heatmap_loss = self.hm_crit(pred_heatmap.sigmoid(), gt_heatmaps)
        losses['loss_heatmap'] = heatmap_loss * self.w_heatmap

        offset_mask = gt_heatmaps.float().repeat_interleave(2, dim=1)
        # print(offset_mask[0, :, 0, 0])
        offset_mask = offset_mask * gt_valid[..., None, None].float()
        self.offset_crit = DenseKLLoss(norm=False, mask_thresh=1.0)
        offset_loss = self.offset_crit(pred_offset, gt_offsets, pred_var, offset_mask)
        
        losses['loss_offset'] = offset_loss * self.w_offset
        return losses


def build_hcr_losses(cfg):
    losses = HCRLosses(cfg)
    return losses
