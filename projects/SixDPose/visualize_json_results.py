#!/usr/bin/env python
# usage: 
'''
python projects/SixDPose/visualize_json_results.py 
--input output/sixdpose/linemod/holepuncher/hcr_rcnn_Rh_50_FPN_1x_hm2_l1loss_hw0p5_ow0p5_tran/inference/coco_instances_results.json 
--output tmp/ --dataset linemod_holepuncher_val
'''

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.structures import BitMasks, Keypoints, PolygonMasks
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.colormap import random_color

from sixdpose import dataset

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass

    try:
        ret.pred_keypoints = [predictions[i]["keypoints"] for i in chosen]
    except KeyError:
        pass
    return ret

class SixDPoseVisualizer(Visualizer):
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions,sem_seg,panoptic_seg_predictions,dataset_dict}`
    that draw composite data in some pre-defined style.

    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according
    to different heuristics, as long as the results still look visually reasonable.
    For example, we currently do not draw class names if there is only one class.
    To obtain a consistent style, implement custom drawing functions with the primitive
    methods instead.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        super().__init__(img_rgb, metadata, scale, instance_mode)

    # def draw_instance_predictions(self, predictions):
    #     """
    #     Draw instance-level prediction results on an image.

    #     Args:
    #         predictions (Instances): the output of an instance detection/segmentation
    #             model. Following fields will be used to draw:
    #             "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

    #     Returns:
    #         output (VisImage): image object with visualizations.
    #     """
    #     boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    #     scores = predictions.scores if predictions.has("scores") else None
    #     classes = predictions.pred_classes if predictions.has("pred_classes") else None
    #     labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
    #     keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    #     if predictions.has("pred_masks"):
    #         masks = np.asarray(predictions.pred_masks)
    #         masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
    #     else:
    #         masks = None

    #     if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
    #         colors = [
    #             self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
    #         ]
    #         alpha = 0.8
    #     else:
    #         colors = None
    #         alpha = 0.5

    #     if self._instance_mode == ColorMode.IMAGE_BW:
    #         self.output.img = self._create_grayscale_image(
    #             (predictions.pred_masks.any(dim=0) > 0).numpy()
    #         )
    #         alpha = 0.3

    #     self.overlay_instances(
    #         masks=masks,
    #         boxes=boxes,
    #         labels=labels,
    #         keypoints=keypoints,
    #         assigned_colors=colors,
    #         alpha=alpha,
    #     )
    #     return self.output

    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            colors = None
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in labels
                ]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            with PathManager.open(dic["sem_seg_file_name"], "rb") as f:
                sem_seg = Image.open(f)
                sem_seg = np.asarray(sem_seg, dtype="uint8")
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                # self.draw_box(boxes[i], edge_color=color)
                pass

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

    # def draw_and_connect_keypoints(self, keypoints):
    #     """
    #     Draws keypoints of an instance and follows the rules for keypoint connections
    #     to draw lines between appropriate keypoints. This follows color heuristics for
    #     line color.

    #     Args:
    #         keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
    #             and the last dimension corresponds to (x, y, probability).

    #     Returns:
    #         output (VisImage): image object with visualizations.
    #     """
    #     visible = {}
    #     keypoint_names = self.metadata.get("keypoint_names")
    #     for idx, keypoint in enumerate(keypoints):
    #         # draw keypoint
    #         x, y, prob = keypoint
    #         if prob > _KEYPOINT_THRESHOLD:
    #             self.draw_circle((x, y), color=_RED)
    #             if keypoint_names:
    #                 keypoint_name = keypoint_names[idx]
    #                 visible[keypoint_name] = (x, y)

    #     if self.metadata.get("keypoint_connection_rules"):
    #         for kp0, kp1, color in self.metadata.keypoint_connection_rules:
    #             if kp0 in visible and kp1 in visible:
    #                 x0, y0 = visible[kp0]
    #                 x1, y1 = visible[kp1]
    #                 color = tuple(x / 255.0 for x in color)
    #                 self.draw_line([x0, x1], [y0, y1], color=color)

    #     # draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
    #     # Note that this strategy is specific to person keypoints.
    #     # For other keypoints, it should just do nothing
    #     try:
    #         ls_x, ls_y = visible["left_shoulder"]
    #         rs_x, rs_y = visible["right_shoulder"]
    #         mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    #     except KeyError:
    #         pass
    #     else:
    #         # draw line from nose to mid-shoulder
    #         nose_x, nose_y = visible.get("nose", (None, None))
    #         if nose_x is not None:
    #             self.draw_line([nose_x, mid_shoulder_x], [nose_y, mid_shoulder_y], color=_RED)

    #         try:
    #             # draw line from mid-shoulder to mid-hip
    #             lh_x, lh_y = visible["left_hip"]
    #             rh_x, rh_y = visible["right_hip"]
    #         except KeyError:
    #             pass
    #         else:
    #             mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
    #             self.draw_line([mid_hip_x, mid_shoulder_x], [mid_hip_y, mid_shoulder_y], color=_RED)
    #     return self.output

    def _convert_keypoints(self, keypoints):
        if isinstance(keypoints, Keypoints):
            keypoints = keypoints.tensor
        keypoints = np.asarray(keypoints).reshape((-1, 9, 5))
        keypoints = keypoints[:, :, [0, 1, 4]]
        return keypoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    # print(metadata)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = SixDPoseVisualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        # vis = Visualizer(img, metadata)
        # vis_gt = vis.draw_dataset_dict(dic).get_image()

        # concat = np.concatenate((vis_pred, vis_gt), axis=1)
        # cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
        cv2.imwrite(os.path.join(args.output, basename), vis_pred[:, :, ::-1])
