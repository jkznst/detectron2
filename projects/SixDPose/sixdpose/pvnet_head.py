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

from .csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer_v3


_TOTAL_SKIPPED = 0

ROI_PVNET_HEAD_REGISTRY = Registry("ROI_PVNET_HEAD")
ROI_PVNET_HEAD_REGISTRY.__doc__ = """
Registry for pvnet keypoint heads, which make keypoint predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_pvnet_head(cfg, input_channels):
    head_name = cfg.MODEL.ROI_PVNET_HEAD.NAME
    return ROI_PVNET_HEAD_REGISTRY.get(head_name)(cfg, input_channels)


@ROI_PVNET_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_channels):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        num_keypoints     = cfg.MODEL.ROI_PVNET_HEAD.NUM_KEYPOINTS
        conv_dims         = cfg.MODEL.ROI_PVNET_HEAD.CONV_HEAD_DIM
        self.norm         = cfg.MODEL.ROI_PVNET_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_PVNET_HEAD.NUM_STACKED_CONVS
        # input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_PVNET_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
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
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.mask_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.vertex_predictor = Conv2d(conv_dims, num_keypoints * 2, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.mask_predictor.weight, std=0.001)
        if self.mask_predictor.bias is not None:
            nn.init.constant_(self.mask_predictor.bias, 0)
        # use normal distribution initialization for vertex prediction layer
        nn.init.normal_(self.vertex_predictor.weight, std=0.001)
        if self.vertex_predictor.bias is not None:
            nn.init.constant_(self.vertex_predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return {"mask": self.mask_predictor(x),
                "vertex": self.vertex_predictor(x)}

@ROI_PVNET_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHead(nn.Module):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, cfg, input_channels):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super(KRCNNConvDeconvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        num_keypoints     = cfg.MODEL.ROI_PVNET_HEAD.NUM_KEYPOINTS
        conv_dims         = cfg.MODEL.ROI_PVNET_HEAD.CONV_HEAD_DIM
        self.norm         = cfg.MODEL.ROI_PVNET_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_PVNET_HEAD.NUM_STACKED_CONVS
        up_scale          = cfg.MODEL.ROI_PVNET_HEAD.UP_SCALE
        deconv_kernel     = cfg.MODEL.ROI_PVNET_HEAD.DECONV_KERNEL
        # input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_PVNET_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []
        for k in range(num_conv):
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
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.mask_lowres = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels, 
            num_mask_classes, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.vertex_lowres = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels, 
            num_keypoints * 2, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        mask_lowres = self.mask_lowres(x)
        vertex_lowres = self.vertex_lowres(x)
        mask = interpolate(mask_lowres, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        vertex = interpolate(vertex_lowres, scale_factor=self.up_scale, mode="bilinear", align_corners=False)

        return {"mask": mask, "vertex": vertex}




class PVNetDataFilter(object):
    def __init__(self, cfg):
        self.iou_threshold = cfg.MODEL.ROI_PVNET_HEAD.FG_IOU_THRESHOLD

    @torch.no_grad()
    def __call__(self, proposals_with_targets):
        """
        Filters proposals with targets to keep only the ones relevant for
        DensePose training
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

def build_pvnet_data_filter(cfg):
    dp_filter = PVNetDataFilter(cfg)
    return dp_filter

@torch.no_grad()
def decode_keypoints(vertexs, mask_probs, rois, mask_prob_threshold=0.5):
    '''
    Arguments:
        vertexs: shape (B, 2*num_kpt, H, W)
        mask_probs: shape (B, 1, H, W)
        rois: shape (B, 4), xyxy
    return:
        keypoints: shape (B, num_pkt, 3)
    '''
    vertexs = vertexs.permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertexs.shape
    vertexs = vertexs.view(b, h, w, vn_2//2, 2)
    masks = (mask_probs > mask_prob_threshold).squeeze(dim=1)
    # print(rois[0])
    kpt_2d = ransac_voting_layer_v3(masks, vertexs, 128, inlier_thresh=0.99, max_num=100)
    # print(kpt_2d[0])

    heatmap_size = h
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    kpt_2d[..., 0] = kpt_2d[..., 0] / scale_x + offset_x
    kpt_2d[..., 1] = kpt_2d[..., 1] / scale_y + offset_y
    kpt_2d = torch.cat((kpt_2d, torch.ones((b, vn_2//2, 1), device=kpt_2d.device)), dim=-1)
    # print(kpt_2d[0])
    return kpt_2d

def pvnet_inference(pvnet_outputs, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pvnet_outputs: {pred_mask_logits, pred_vertex}
                pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
                pred_vertex (Tensor):
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    pred_mask_logits, pred_vertex = pvnet_outputs["mask"], pvnet_outputs["vertex"] 
    
    if pred_mask_logits.size(0) > 0:
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits.sigmoid()
        else:
            # Select masks corresponding to the predicted classes
            num_masks = pred_mask_logits.shape[0]
            class_pred = cat([i.pred_classes for i in pred_instances])
            indices = torch.arange(num_masks, device=class_pred.device)
            mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

        num_boxes_per_image = [len(i) for i in pred_instances]
        # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
        bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)
        keypoint_results = decode_keypoints(pred_vertex.detach(), mask_probs_pred.detach(), bboxes_flat.detach())

        mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
        # keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_boxes_per_image, dim=0)
        keypoint_results = keypoint_results.split(num_boxes_per_image, dim=0)

        for prob, instances in zip(mask_probs_pred, pred_instances):
            instances.pred_masks = prob  # (num_instance_per_img, 1, Hmask, Wmask)

        for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
            # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y)
            instances_per_image.pred_keypoints = keypoint_results_per_image
    # elif pred_mask_logits.size(0) == 0:
    #     for instances in pred_instances:
    #         instances.pred_masks = torch.zeros(size=(0, 0, 0, 0), device=pred_mask_logits.device)
    #         instances.pred_keypoints = torch.zeros(size=(0, 0, 0), device=pred_vertex.device)

def compute_vertex(masks, kpt_2d):
    '''
    arguments:
        masks: shape (N, H, W)
        kpt_2d: shape (N, K, 2), keypoint coordinates w.r.t mask rois
    return:
        vertex: shape (N, H, W, K * 2)
    '''
    
    assert masks.size(0) == kpt_2d.size(0), "batch size does not match !"
    n = masks.size(0)
    h, w = masks.size(1), masks.size(2)
    num_kpt = kpt_2d.size(1)
    x = torch.arange(w).repeat(h, 1)
    y = torch.arange(h).repeat(w, 1).transpose(0, 1)
    xy = torch.stack([x, y], dim=-1).reshape(1, h, w, 1, 2).to(device=kpt_2d.device)

    vertex = kpt_2d.reshape(n, 1, 1, num_kpt, 2) - xy   # shape (N, H, W, K, 2)
    norm = torch.norm(vertex, dim=-1, keepdim=True)
    norm[norm < 1e-3] += 1e-3
    vertex = vertex / norm

    vertex_out = torch.zeros_like(vertex)
    vertex_out = torch.where(masks.reshape(n, h, w, 1, 1), 
        vertex, vertex_out)
    # print(masks[0])
    # print(vertex_out[0, 0, :, :, 0])
    
    return vertex_out.reshape(n, h, w, num_kpt * 2)

@torch.no_grad()
def keypoints_to_vertex(
    keypoints: torch.Tensor, rois: torch.Tensor, masks: torch.Tensor, heatmap_size: int):
    """
    Encode keypoint locations into a target pixel-wise vertex for use in pvnet.

    Arguments:
        keypoints: tensor of keypoint locations in of shape (N, K, 3). float
        rois: Nx4 tensor of rois in xyxy format
        masks: tensor of mask of shape (N, H, W)
        heatmap_size: integer side length of square heatmap.

    Returns:
        vertexs: A tensor of shape (N, K * 2, H, W) encoding keypoint coordinates.
        valid: A tensor of shape (N, K * 2) containing whether each keypoint is in
            the roi or not.
    """
    assert masks.size(1) == masks.size(2) == heatmap_size, "mask shape does not match!"
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    # x = x.floor().long()
    y = (y - offset_y) * scale_y
    # y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    vertexs = compute_vertex(masks, torch.stack([x, y], dim=-1))
    vertexs = vertexs.permute(0, 3, 1, 2)

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

    # valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    # valid = (valid_loc & vis).long()
    valid = vis.long()
    valid = valid.repeat(1, 2).reshape(valid.size(0), 2, -1).permute(0, 2, 1).reshape(valid.size(0), -1)

    # print(keypoints)
    # print(vertexs.shape, valid.shape)
    # print(valid)
    # print(vertexs.device, valid.device)

    return vertexs, valid

class PVNetLosses(object):
    def __init__(self, cfg):
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_PVNET_HEAD.HEATMAP_SIZE
        self.w_mask       = cfg.MODEL.ROI_PVNET_HEAD.MASK_WEIGHTS
        self.w_vertex     = cfg.MODEL.ROI_PVNET_HEAD.VERTEX_WEIGHTS
        # fmt: on

    def __call__(self, proposals_with_gt, pvnet_outputs, vis_period=0):
        """
        Compute the mask prediction loss defined in the Mask R-CNN paper.

        Args:
            pvnet_outputs: {pred_mask_logits, pred_vertex}
                pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
                pred_vertex (Tensor):
            proposals_with_gt (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1 correspondence with the pred_mask_logits. 
                The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.

        Returns:
            mask_loss (Tensor): A scalar tensor containing the loss.
        """
        losses = {}
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        pred_mask_logits, pred_vertex = pvnet_outputs["mask"], pvnet_outputs["vertex"]
        
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        assert self.heatmap_size == pred_mask_logits.size(2)
        mask_side_len = self.heatmap_size
        assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_masks = []
        gt_vertexs = []
        gt_valid = []
        for instances_per_image in proposals_with_gt:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

            # compute vertex
            gt_vertexs_per_image, gt_valid_per_image = keypoints_to_vertex(instances_per_image.gt_keypoints.tensor,
                instances_per_image.proposal_boxes.tensor, gt_masks_per_image, mask_side_len
            )
            gt_vertexs_per_image = gt_vertexs_per_image.to(device=pred_vertex.device)
            gt_valid_per_image = gt_valid_per_image.to(device=pred_vertex.device)
            gt_vertexs.append(gt_vertexs_per_image)
            gt_valid.append(gt_valid_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0) # shape (N, H, W)
        gt_vertexs = cat(gt_vertexs, dim=0) # shape (N, 2K, H, W)
        gt_valid = cat(gt_valid, dim=0) # shape (N, 2K)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)
        gt_vertexs = gt_vertexs.to(dtype=torch.float32)
        gt_valid = gt_valid.to(dtype=torch.float32)

        # Log the training accuracy (using gt classes and 0.5 threshold)
        mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = gt_masks_bool.sum().item()
        false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
            gt_masks_bool.numel() - num_positive, 1.0
        )
        false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

        storage = get_event_storage()
        storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
        storage.put_scalar("mask_rcnn/false_positive", false_positive)
        storage.put_scalar("mask_rcnn/false_negative", false_negative)
        if vis_period > 0 and storage.iter % vis_period == 0:
            pred_masks = pred_mask_logits.sigmoid()
            vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
            name = "Left: mask prediction;   Right: mask GT"
            for idx, vis_mask in enumerate(vis_masks):
                vis_mask = torch.stack([vis_mask] * 3, axis=0)
                storage.put_image(name + f" ({idx})", vis_mask)

        mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
        losses['loss_mask'] = mask_loss * self.w_mask

        # gt_mask shape is (N, H, W), weight shape is (N, 1, H, W)
        weight = gt_masks[:, None].float()
        weight = weight * gt_valid[..., None, None].float()
        self.vote_crit = F.smooth_l1_loss
        vote_loss = self.vote_crit(pred_vertex * weight, gt_vertexs * weight, reduction='sum')
        # normalize vote loss
        vote_loss = vote_loss / (weight.sum() + 1e-3)
        losses['loss_vertex'] = vote_loss * self.w_vertex
        return losses


def build_pvnet_losses(cfg):
    losses = PVNetLosses(cfg)
    return losses
