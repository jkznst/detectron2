# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
from typing import List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

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
        self.anchor_matcher = Matcher(
            cfg.MODEL.CRPNET.IOU_THRESHOLDS,
            cfg.MODEL.CRPNET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        return self.pixel_mean.device

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
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
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

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        # print(features[0].size())
        
        anchors = self.anchor_generator(features)
        # print(anchors[0].tensor.size())
        pred_logits, pred_anchor_deltas, pred_kpt_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]
        pred_kpt_deltas = [permute_to_N_HWA_K(x, self.num_kpt * 2) for x in pred_kpt_deltas]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes, gt_keypoints = self.label_anchors(anchors, gt_instances)
            # print(gt_classes.size())
            # print(gt_anchors_reg_deltas.size())
            # print(gt_kpt_reg_deltas.size())
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes,
                    pred_kpt_deltas, gt_keypoints)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, pred_kpt_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, pred_kpt_deltas, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes,
            pred_kpt_deltas, gt_keypoints):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes, gt_keypoints: see output of :meth:`CRPNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4) and (N, R, num_kpt x 2), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas, pred_kpt_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4 or num_kpt x 2)

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        if self.cascade_regression:
            # predicted_boxes = [self.box2box_transform.apply_deltas(d, anchors) for d in cat(pred_anchor_deltas, dim=1)]
            # predicted_boxes = torch.stack(predicted_boxes)
            # TODO: test if we should use gt bbox or pred bbox
            gt_kpt_deltas = [self.box2kpt_transform.get_deltas(b, k) for b, k in zip(gt_boxes, gt_keypoints)]
            # print(gt_kpt_reg_deltas_i[0])
            # test_kpt = self.box2kpt_transform.apply_deltas(
            #     gt_kpt_reg_deltas_i, matched_gt_boxes.tensor
            # )
            # print(test_kpt[0])
        else:
            gt_kpt_deltas = [self.box2kpt_transform.get_deltas(anchors, k) for k in gt_keypoints]
        gt_kpt_deltas = torch.stack(gt_kpt_deltas)  # (N, R, num_kpt x 2)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        )

        loss_kpt_reg = smooth_l1_loss(
            cat(pred_kpt_deltas, dim=1)[pos_mask],
            gt_kpt_deltas[pos_mask],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        )
        return {
            "loss_cls": loss_cls / self.loss_normalizer,
            "loss_box_reg": loss_box_reg / self.loss_normalizer,
            "loss_kpt_reg": loss_kpt_reg / self.loss_normalizer / self.num_kpt * self.kpt_loss_weight,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            gt_labels: list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            matched_gt_boxes (list[Tensor]):
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
            matched_gt_kpts (list[Tensor]):
                i-th element is a Rx(num_kptx2) tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt keypoints for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        anchors = Boxes.cat(anchors)  # Rx4
        
        gt_labels = []
        matched_gt_boxes = []
        matched_gt_kpts = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                matched_gt_kpts_i = gt_per_image.gt_keypoints.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                matched_gt_kpts_i = torch.zeros(anchors.tensor.size(0), self.num_kpt * 2)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_gt_kpts.append(matched_gt_kpts_i)

        return gt_labels, matched_gt_boxes, matched_gt_kpts
        
    def inference(self, anchors, pred_logits, pred_anchor_deltas, pred_kpt_deltas, image_sizes):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            bbox_deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            kpt_deltas_per_image = [x[img_idx] for x in pred_kpt_deltas]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, bbox_deltas_per_image, 
                kpt_deltas_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors, box_cls, box_delta, kpt_delta, image_size):
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
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
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
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                groups=in_channels))
            cls_subnet.append(nn.ReLU())
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1))
            cls_subnet.append(nn.ReLU())

            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                groups=in_channels))
            bbox_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1))
            bbox_subnet.append(nn.ReLU())

            kpt_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                groups=in_channels))
            kpt_subnet.append(nn.ReLU())
            kpt_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1))
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


class SSDHead(nn.Module):
    """
    The head used in SSD for object classification and box regression.
    Does not share parameters across feature levels.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels      = [f.channels for f in input_shape]
        num_classes      = cfg.MODEL.RETINANET.NUM_CLASSES
        prior_prob       = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        for i, in_channel in enumerate(in_channels):
            cls_score = nn.Conv2d(
                in_channel, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
                )
            torch.nn.init.normal_(cls_score.weight, mean=0, std=0.01)
            torch.nn.init.constant_(cls_score.bias, bias_value)
            self.add_module("p{}_cls_score".format(i + 3), cls_score)

            bbox_pred = nn.Conv2d(
                in_channel, num_anchors * 4, kernel_size=3, stride=1, padding=1
                )
            torch.nn.init.normal_(bbox_pred.weight, mean=0, std=0.01)
            torch.nn.init.constant_(bbox_pred.bias, 0)
            self.add_module("p{}_bbox_pred".format(i + 3), bbox_pred)
            

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for i, feature in enumerate(features):
            cls_score = getattr(self, "p{}_cls_score".format(i + 3))
            bbox_pred = getattr(self, "p{}_bbox_pred".format(i + 3))
            logits.append(cls_score(feature))
            bbox_reg.append(bbox_pred(feature))
        return logits, bbox_reg
