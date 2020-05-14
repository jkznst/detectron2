# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import copy
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from fvcore.common.file_io import PathManager
from fvcore.transforms.transform import NoOpTransform, Transform

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .dataset import BB8_KEYPOINT_CONNECTION_RULES, FPS8_KEYPOINT_CONNECTION_RULES
# from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData

class RandomBlurTransform(Transform):
    def __init__(self, blur_sigma=1):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply blur transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blur would not
                require interpolation.
        Returns:
            ndarray: blured image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = cv2.GaussianBlur(img, (self.blur_sigma, self.blur_sigma), 0)
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return cv2.GaussianBlur(img, (self.blur_sigma, self.blur_sigma), 0)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

class ColorJitterTransform(Transform):
    def __init__(self, brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply color jitter transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform color jitter would not
                require interpolation.
        Returns:
            ndarray: color jittered image(s).
        """
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue)
        img = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(img, np.uint8))))
        return img
        
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

class RandomBlur(T.TransformGen):
    """
    Randomly gussian blur an image.
    """
    def __init__(self, blur_prob=0.5, blur_sigma=None):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        do = self._rand_range() < self.blur_prob
        if do:
            if self.blur_sigma is None:
                self.blur_sigma = np.random.choice([3, 5, 7, 9])
            return RandomBlurTransform(self.blur_sigma)
        else:
            return NoOpTransform()

class ColorJitter(T.TransformGen):
    """
    Color jitter an image.
    """
    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        return ColorJitterTransform(self.brightness, self.contrast, self.saturation, self.hue)

def create_sixdpose_keypoint_hflip_indices(dataset_names, keypoint_format):
    """
    Args:
        dataset_names (list[str]): list of dataset names
        keypoint_format(str): bb8, fps8, or bb8+fps8
    Returns:
        ndarray[int]: a vector of size=#keypoints, storing the
        horizontally-flipped keypoint indices.
    """
    meta = MetadataCatalog.get(dataset_names[0])
    keypoint_flip_map = ()  # sixd pose has no filp map

    if keypoint_format == 'bb8':
        names = (
        "center",
        # bb8
        "bb8_0", "bb8_1",
        "bb8_2", "bb8_3",
        "bb8_4", "bb8_5",
        "bb8_6", "bb8_7",
        )
        connection_rules = BB8_KEYPOINT_CONNECTION_RULES
        meta.set(keypoint_names=names, keypoint_flip_map=keypoint_flip_map, keypoint_connection_rules=connection_rules)
    elif keypoint_format == 'fps8':
        names = (
        "center",
        # fps8
        "fps8_0", "fps8_1",
        "fps8_2", "fps8_3",
        "fps8_4", "fps8_5",
        "fps8_6", "fps8_7",
        )
        connection_rules = FPS8_KEYPOINT_CONNECTION_RULES
        meta.set(keypoint_names=names, keypoint_flip_map=keypoint_flip_map, keypoint_connection_rules=connection_rules)
    else:
        assert keypoint_format == 'bb8+fps8', keypoint_format
        names = (
        "center",
        # bb8
        "bb8_0", "bb8_1",
        "bb8_2", "bb8_3",
        "bb8_4", "bb8_5",
        "bb8_6", "bb8_7",
        # fps8
        "fps8_0", "fps8_1",
        "fps8_2", "fps8_3",
        "fps8_4", "fps8_5",
        "fps8_6", "fps8_7",
        )
        connection_rules = BB8_KEYPOINT_CONNECTION_RULES + FPS8_KEYPOINT_CONNECTION_RULES
        meta.set(keypoint_names=names, keypoint_flip_map=keypoint_flip_map, keypoint_connection_rules=connection_rules)
    
    # TODO flip -> hflip 
    flip_map = dict(keypoint_flip_map)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices)


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        if cfg.INPUT.RANDOMBLUR.ENABLED and is_train:
            self.blur_gen = RandomBlur(cfg.INPUT.RANDOMBLUR.PROB)
            logging.getLogger(__name__).info("BlurGen used in training: " + str(self.blur_gen))
        else:
            self.blur_gen = None 

        if cfg.INPUT.COLORJITTER.ENABLED and is_train:
            self.colorjitter_gen = ColorJitter(cfg.INPUT.COLORJITTER.BRIGHTNESS, cfg.INPUT.COLORJITTER.CONTRAST,
                                                cfg.INPUT.COLORJITTER.SATURATION, cfg.INPUT.COLORJITTER.HUE)
            logging.getLogger(__name__).info("ColorJitterGen used in training: " + str(self.colorjitter_gen))
        else:
            self.colorjitter_gen = None 

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON or cfg.MODEL.PVNET_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON or cfg.MODEL.PVNET_ON
        self.keypoint_format= cfg.INPUT.KEYPOINT_FORMAT
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = create_sixdpose_keypoint_hflip_indices(cfg.DATASETS.TRAIN, self.keypoint_format)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + 
                ([self.blur_gen] if self.blur_gen else []) + 
                ([self.colorjitter_gen] if self.colorjitter_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            if self.blur_gen:
                blur_tfm = self.blur_gen.get_transform(image)
                image = blur_tfm.apply_image(image)
            if self.colorjitter_gen:
                colorjitter_tfm = self.colorjitter_gen.get_transform(image)
                image = colorjitter_tfm.apply_image(image)

            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.colorjitter_gen:
                transforms = colorjitter_tfm + transforms
            if self.blur_gen:
                transforms = blur_tfm + transforms
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        ).contiguous()
        # Can use uint8 if it turns out to be slow some day

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)
                # USER: load keypoints according to keypoint_format
                else:
                    keypts = anno["keypoints"]
                    if 'bb8' in self.keypoint_format:
                        corner_2d = np.array(anno["corner_2d"])
                        corner_2d = np.insert(corner_2d, 2, 2, axis=1).flatten().tolist()
                        keypts += corner_2d
                    if 'fps8' in self.keypoint_format:
                        fps_2d = np.array(anno["fps_2d"])
                        fps_2d = np.insert(fps_2d, 2, 2, axis=1).flatten().tolist()
                        keypts += fps_2d
                    anno["keypoints"] = keypts

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        # if "sem_seg_file_name" in dataset_dict:
        #     with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
        #         sem_seg_gt = Image.open(f)
        #         sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
        #     sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
        #     sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        #     dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

class COCODatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON or cfg.MODEL.PVNET_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON or cfg.MODEL.PVNET_ON or cfg.MODEL.CRPNET_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict