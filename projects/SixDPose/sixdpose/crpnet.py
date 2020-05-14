# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
from typing import List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess, build_backbone, build_anchor_generator

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher

__all__ = ["CRPNet"]


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, kpt_delta, 
        num_classes=80, num_kpt=17):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta and the kpt_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    kpt_delta_flattened = [permute_to_N_HWA_K(x, num_kpt * 2) for x in kpt_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    kpt_delta = cat(kpt_delta_flattened, dim=1).view(-1, num_kpt * 2)
    return box_cls, box_delta, kpt_delta


# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)

@torch.jit.script
class Box2KptTransform(object):
    """
    The box-to-kpt transform defined in CRPNet. The transformation is parameterized
    by 2 deltas: (dx, dy). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_kpts):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_kpts`. That is, the relation
        ``target_kpts == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals, shape (N, 4)
            target_kpts (Tensor): target of the transformation, e.g., ground-truth
                keypoints. shape (N, K, 3)
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_kpts, torch.Tensor), type(target_kpts)
        # print(src_boxes[0:10])
        # print(target_kpts[0:10])

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        # target_widths = target_kpts[:, 2] - target_kpts[:, 0]
        # target_heights = target_kpts[:, 3] - target_kpts[:, 1]
        target_kpt_x = target_kpts[..., 0]
        target_kpt_y = target_kpts[..., 1]

        wx, wy, ww, wh = self.weights
        # print(self.weights)
        # print(target_kpt_x[0])
        # print(src_ctr_x[0])
        # print(src_widths[0])
        dx = wx * (target_kpt_x - src_ctr_x[:, None]) / src_widths[:, None]
        dy = wy * (target_kpt_y - src_ctr_y[:, None]) / src_heights[:, None]
        # dw = ww * torch.log(target_widths / src_widths)
        # dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy), dim=2).reshape(target_kpts.size(0), -1)  # shape (N, K x 2)
        # print(deltas.size())
        assert (src_widths > 0).all().item(), "Input boxes to Box2KptTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, A*k*2), where A >= 1.
                deltas[i] represents A potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        return:
            keypoints of shape (N, K, 3)
        """
        boxes = boxes.to(deltas.dtype)
        # print(deltas.size())
        # print(boxes.size())

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::2] / wx
        dy = deltas[:, 1::2] / wy
        # dw = deltas[:, 2::4] / ww
        # dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        # dw = torch.clamp(dw, max=self.scale_clamp)
        # dh = torch.clamp(dh, max=self.scale_clamp)

        pred_kpt_x = dx * widths[:, None] + ctr_x[:, None]
        pred_kpt_y = dy * heights[:, None] + ctr_y[:, None]
        # pred_w = torch.exp(dw) * widths[:, None]
        # pred_h = torch.exp(dh) * heights[:, None]

        kpt_score = torch.ones((deltas.size(0), deltas.size(1) // 2), device=deltas.device)
        pred_kpts = torch.stack((pred_kpt_x, pred_kpt_y, kpt_score), dim=2)
        return pred_kpts


@META_ARCH_REGISTRY.register()
class CRPNet(nn.Module):
    """
    Implement CRPNet.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes              = cfg.MODEL.CRPNET.NUM_CLASSES
        self.in_features              = cfg.MODEL.CRPNET.IN_FEATURES
        self.num_kpt                  = cfg.MODEL.CRPNET.NUM_KEYPOINTS
        self.cascade_regression       = cfg.MODEL.CRPNET.CASCADE_REGRESSION
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.CRPNET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.CRPNET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.CRPNET.SMOOTH_L1_LOSS_BETA
        self.kpt_loss_weight          = cfg.MODEL.CRPNET.KPT_WEIGHT
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.CRPNET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.CRPNET.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.CRPNET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period               = cfg.VIS_PERIOD
        self.input_format             = cfg.INPUT.FORMAT
        # fmt: on

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = CRPNetHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.CRPNET.BBOX_REG_WEIGHTS)
        self.box2kpt_transform = Box2KptTransform(weights=cfg.MODEL.CRPNET.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.CRPNET.IOU_THRESHOLDS,
            cfg.MODEL.CRPNET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.input_format == "BGR":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta, kpt_delta = self.head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas, gt_kpt_reg_deltas = \
                self.get_ground_truth(anchors, gt_instances, box_delta)
            # print(gt_classes.size())
            # print(gt_anchors_reg_deltas.size())
            # print(gt_kpt_reg_deltas.size())
            losses = self.losses(gt_classes, gt_anchors_reg_deltas, gt_kpt_reg_deltas,
                 box_cls, box_delta, kpt_delta)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(box_cls, box_delta, kpt_delta, anchors, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(box_cls, box_delta, kpt_delta, anchors, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_classes, gt_anchors_deltas, gt_kpt_deltas,
             pred_class_logits, pred_anchor_deltas, pred_kpt_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`CRPNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas, pred_kpt_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, pred_kpt_deltas, self.num_classes, self.num_kpt
        )  # Shapes: (N x R, C) and (N x R, 4) and (NxR, Kx2), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)
        gt_kpt_deltas = gt_kpt_deltas.view(-1, self.num_kpt * 2)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum().item()
        # print(num_foreground)
        get_event_storage().put_scalar("num_foreground", num_foreground)
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * num_foreground
        )

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
        
        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        # regression loss
        # print(pred_anchor_deltas.size())
        # print(gt_anchors_deltas.size())
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        # print(pred_kpt_deltas.size())
        # print(gt_kpt_deltas.size())
        loss_kpt_reg = smooth_l1_loss(
            pred_kpt_deltas[foreground_idxs],
            gt_kpt_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, self.loss_normalizer) / self.num_kpt * self.kpt_loss_weight
        # print(loss_cls, loss_box_reg, loss_kpt_reg)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg, "loss_kpt_reg": 0.0}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets, pred_bbox_delta):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, C-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "C". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
            gt_anchors_kpt_deltas (Tensor):
                Shape (N, R, Kx2).
                The last dimension represents ground-truth box2kpt transform
                targets (dx, dy) that map each anchor to its matched ground-truth keypoint.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        gt_classes = []
        gt_anchors_deltas = []
        gt_kpt_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)
            # print(gt_matched_idxs.size())
            # print(gt_matched_idxs[0:10])
            # print(anchor_labels[0:10])
            # print(targets_per_image)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_per_image.tensor, matched_gt_boxes.tensor
                )

                # ground truth keypoint regression
                matched_gt_kpts = targets_per_image.gt_keypoints[gt_matched_idxs]
                if self.cascade_regression:
                    # TODO: test if we should use gt bbox or pred bbox
                    gt_kpt_reg_deltas_i = self.box2kpt_transform.get_deltas(
                        matched_gt_boxes.tensor, matched_gt_kpts.tensor
                    )
                else:
                    gt_kpt_reg_deltas_i = self.box2kpt_transform.get_deltas(
                        anchors_per_image.tensor, matched_gt_kpts.tensor
                    )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)
                gt_kpt_reg_deltas_i = torch.zeros(anchors_per_image.tensor.size(0), self.num_kpt * 2)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
            gt_kpt_deltas.append(gt_kpt_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas), torch.stack(gt_kpt_deltas)

    def inference(self, box_cls, box_delta, kpt_delta, anchors, image_sizes):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`CRPNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(image_sizes)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        kpt_delta = [permute_to_N_HWA_K(x, self.num_kpt * 2) for x in kpt_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            kpt_reg_per_image = [kpt_reg_per_level[img_idx] for kpt_reg_per_level in kpt_delta]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, kpt_reg_per_image, anchors_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, kpt_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, C)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that C becomes 4.
            kpt_delta (list[Tensor]): Same shape as 'box_delta' except that 4 becomes K x 2.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        kpts_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, kpt_reg_i, anchors_i in zip(box_cls, box_delta, kpt_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            kpt_reg_i = kpt_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)
            if self.cascade_regression:
                predicted_kpts = self.box2kpt_transform.apply_deltas(kpt_reg_i, predicted_boxes)
            else:
                predicted_kpts = self.box2kpt_transform.apply_deltas(kpt_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            kpts_all.append(predicted_kpts)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, kpts_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, kpts_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        result.pred_keypoints = kpts_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class CRPNetHead(nn.Module):
    """
    The head used in CRPNet for object classification and box regression and pose estimation.
    It has three subnets for the three tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = input_shape[0].channels
        num_classes      = cfg.MODEL.CRPNET.NUM_CLASSES
        num_convs        = cfg.MODEL.CRPNET.NUM_CONVS
        prior_prob       = cfg.MODEL.CRPNET.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        num_kpt          = cfg.MODEL.CRPNET.NUM_KEYPOINTS
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        kpt_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU())
            kpt_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            kpt_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.kpt_subnet = nn.Sequential(*kpt_subnet)
        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.kpt_pred = nn.Conv2d(in_channels, num_anchors * num_kpt * 2, 
            kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.kpt_subnet, self.cls_score, self.bbox_pred, self.kpt_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxC, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and C object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
            kpt_reg (list[Tensor]): #lvl tensors, each has shape (N, AxKx2, Hi, Wi).
                The tensor predicts Kx2-vector (dx,dy) keypoint
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth keypoint.
        """
        logits = []
        bbox_reg = []
        kpt_reg = []
        for feature in features:
            # print(feature.size())
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
            kpt_reg.append(self.kpt_pred(self.kpt_subnet(feature)))
        return logits, bbox_reg, kpt_reg
