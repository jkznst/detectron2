# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import contextlib
import os
import cv2
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

logger = logging.getLogger(__name__)

# fmt: off
SIXDPOSE_KEYPOINT_NAMES = (
    "center",
    # bb8
    "bb8_0", "bb8_1",
    "bb8_2", "bb8_3",
    "bb8_4", "bb8_5",
    "bb8_6", "bb8_7",
    # fps8
    # "fps8_0", "fps8_1",
    # "fps8_2", "fps8_3",
    # "fps8_4", "fps8_5",
    # "fps8_6", "fps8_7",
)
# fmt: on

# Pairs of keypoints that should be exchanged under horizontal flipping
# SIXDPOSE_KEYPOINT_FLIP_MAP = (
#     # ("left_eye", "right_eye"),
#     # ("left_ear", "right_ear"),
#     # ("left_shoulder", "right_shoulder"),
#     # ("left_elbow", "right_elbow"),
#     # ("left_wrist", "right_wrist"),
#     # ("left_hip", "right_hip"),
#     # ("left_knee", "right_knee"),
#     # ("left_ankle", "right_ankle"),
# )

# rules for pairs of keypoints to draw a line between, and the line color to use.
BB8_KEYPOINT_CONNECTION_RULES = [
    # bb8
    ("bb8_0", "bb8_1", (102, 204, 255)),
    ("bb8_1", "bb8_2", (102, 204, 255)),
    ("bb8_2", "bb8_3", (102, 204, 255)),
    ("bb8_3", "bb8_0", (102, 204, 255)),
    ("bb8_4", "bb8_5", (51, 153, 255)),
    ("bb8_5", "bb8_6", (51, 153, 255)),
    ("bb8_6", "bb8_7", (51, 153, 255)),
    ("bb8_7", "bb8_4", (51, 153, 255)),
    ("bb8_0", "bb8_4", (102, 0, 204)),
    ("bb8_1", "bb8_5", (102, 0, 204)),
    ("bb8_2", "bb8_6", (102, 0, 204)),
    ("bb8_3", "bb8_7", (102, 0, 204)),
]

FPS8_KEYPOINT_CONNECTION_RULES = [
    # fps8
    ("center", "fps8_0", (255, 128, 0)),
    ("center", "fps8_1", (153, 255, 204)),
    ("center", "fps8_2", (128, 229, 255)),
    ("center", "fps8_3", (153, 255, 153)),
    ("center", "fps8_4", (102, 255, 224)),
    ("center", "fps8_5", (255, 102, 0)),
    ("center", "fps8_6", (255, 255, 77)),
    ("center", "fps8_7", (153, 255, 204)),
]

def get_sixdpose_metadata():
    meta = {
        "keypoint_names": SIXDPOSE_KEYPOINT_NAMES,
        # "keypoint_flip_map": SIXDPOSE_KEYPOINT_FLIP_MAP,
        "keypoint_connection_rules": BB8_KEYPOINT_CONNECTION_RULES,
    }
    return meta

def load_occlusion_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO
    import pycocotools.mask as mask_util

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map
        print(meta)

    # sort indices for reproducible results
    img_ids = sorted(list(coco_api.imgs.keys()))
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    # imgs = coco_api.loadImgs(img_ids)
    # print(imgs[0])
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    # anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    keep_idx = []
    anns = []
    for img_id in img_ids:
        ann = coco_api.imgToAnns[img_id]
        if len(ann) > 0:
            anns.append(ann)
            keep_idx.append(img_id)
    imgs = coco_api.loadImgs(keep_idx)


    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    print("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["bbox", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        record["sem_seg_file_name"] = os.path.join(image_root, img_dict["seg_map"])

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:
                if isinstance(segm, str):   # path
                    mask_path = os.path.join(image_root, segm)  # binary mask
                    if not os.path.exists(mask_path):
                        num_instances_without_valid_segmentation += 1
                        continue
                    segm = cv2.imread(mask_path, 0)
                    # cv2.imshow('1', segm)
                    # cv2.waitKey(0)
                    segm = np.asfortranarray(segm)
                    segm = mask_util.encode(segm)
                    # print(segm)
                obj["segmentation"] = segm

            segm_occagn = anno.get("segmentation_occagn", None)
            if segm_occagn:
                if isinstance(segm_occagn, str):
                    mask_occagn_path = os.path.join(image_root, segm_occagn)  # binary mask
                    if not os.path.exists(mask_occagn_path):
                        continue
                    # segm_occagn = cv2.imread(mask_occagn_path)
                    # # cv2.imshow('1', segm_occagn)
                    # # cv2.waitKey(0)
                    # segm_occagn = np.asfortranarray(segm_occagn)
                    # segm_occagn = mask_util.encode(segm_occagn)
                obj["segmentation_occagn"] = segm_occagn

            # USER: we only load center here, load other kpts in DatasetMapper
            keypts = anno.get("center_2d", None)
            if keypts:
                keypts.append(2)
            obj["keypoints"] = keypts
            # if keypts:  # list[list[float]]
            #     keypts = np.array(keypts)
            #     keypts = np.insert(keypts, 2, 2, axis=1).flatten().tolist()
            #     obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


SPLITS = {
    # occlusion dataset of all 8 objects
    "occlusion_train": ("occlusion", "occlusion/occlusion_train.json"),
    "occlusion_val": ("occlusion", "occlusion/occlusion_val.json"),
    # occlusion dataset of single object
    "occlusion_ape_train": ("occlusion", "occlusion/occlusion_ape_train.json"),
    "occlusion_ape_val": ("occlusion", "occlusion/occlusion_ape_val.json"),
    "occlusion_can_train": ("occlusion", "occlusion/occlusion_can_train.json"),
    "occlusion_can_val": ("occlusion", "occlusion/occlusion_can_val.json"),
    "occlusion_cat_train": ("occlusion", "occlusion/occlusion_cat_train.json"),
    "occlusion_cat_val": ("occlusion", "occlusion/occlusion_cat_val.json"),
    "occlusion_driller_train": ("occlusion", "occlusion/occlusion_driller_train.json"),
    "occlusion_driller_val": ("occlusion", "occlusion/occlusion_driller_val.json"),
    "occlusion_duck_train": ("occlusion", "occlusion/occlusion_duck_train.json"),
    "occlusion_duck_val": ("occlusion", "occlusion/occlusion_duck_val.json"),
    "occlusion_eggbox_train": ("occlusion", "occlusion/occlusion_eggbox_train.json"),
    "occlusion_eggbox_val": ("occlusion", "occlusion/occlusion_eggbox_val.json"),
    "occlusion_glue_train": ("occlusion", "occlusion/occlusion_glue_train.json"),
    "occlusion_glue_val": ("occlusion", "occlusion/occlusion_glue_val.json"),
    "occlusion_holepuncher_train": ("occlusion", "occlusion/occlusion_holepuncher_train.json"),
    "occlusion_pbr_holepuncher_train": ("occlusion_pbr", "occlusion_pbr/occlusion_pbr_holepuncher_train.json"),
    "occlusion_holepuncher_val": ("occlusion", "occlusion/occlusion_holepuncher_val.json"),
    # linemod datasets
    "linemod_ape_train": ("linemod/ape", "linemod/ape/linemod_ape_train.json"),
    "linemod_ape_val": ("linemod/ape", "linemod/ape/linemod_ape_val.json"),
    "linemod_benchvise_train": ("linemod/benchvise", "linemod/benchvise/linemod_benchvise_train.json"),
    "linemod_benchvise_val": ("linemod/benchvise", "linemod/benchvise/linemod_benchvise_val.json"),
    "linemod_cam_train": ("linemod/cam", "linemod/cam/linemod_cam_train.json"),
    "linemod_cam_val": ("linemod/cam", "linemod/cam/linemod_cam_val.json"),
    "linemod_can_train": ("linemod/can", "linemod/can/linemod_can_train.json"),
    "linemod_can_val": ("linemod/can", "linemod/can/linemod_can_val.json"),
    "linemod_cat_train": ("linemod/cat", "linemod/cat/linemod_cat_train.json"),
    "linemod_cat_val": ("linemod/cat", "linemod/cat/linemod_cat_val.json"),
    "linemod_driller_train": ("linemod/driller", "linemod/driller/linemod_driller_train.json"),
    "linemod_driller_val": ("linemod/driller", "linemod/driller/linemod_driller_val.json"),
    "linemod_duck_train": ("linemod/duck", "linemod/duck/linemod_duck_train.json"),
    "linemod_duck_val": ("linemod/duck", "linemod/duck/linemod_duck_val.json"),
    "linemod_eggbox_train": ("linemod/eggbox", "linemod/eggbox/linemod_eggbox_train.json"),
    "linemod_eggbox_val": ("linemod/eggbox", "linemod/eggbox/linemod_eggbox_val.json"),
    "linemod_glue_train": ("linemod/glue", "linemod/glue/linemod_glue_train.json"),
    "linemod_glue_val": ("linemod/glue", "linemod/glue/linemod_glue_val.json"),
    "linemod_holepuncher_train": ("linemod/holepuncher", "linemod/holepuncher/linemod_holepuncher_train.json"),
    "linemod_holepuncher_val": ("linemod/holepuncher", "linemod/holepuncher/linemod_holepuncher_val.json"),
    "linemod_iron_train": ("linemod/iron", "linemod/iron/linemod_iron_train.json"),
    "linemod_iron_val": ("linemod/iron", "linemod/iron/linemod_iron_val.json"),
    "linemod_lamp_train": ("linemod/lamp", "linemod/lamp/linemod_lamp_train.json"),
    "linemod_lamp_val": ("linemod/lamp", "linemod/lamp/linemod_lamp_val.json"),
    "linemod_phone_train": ("linemod/phone", "linemod/phone/linemod_phone_train.json"),
    "linemod_phone_val": ("linemod/phone", "linemod/phone/linemod_phone_val.json"),
    # tless datasets TODO
    "tless_toy_05_train": ("tless_toy/obj_05", "tless_toy/obj_05/train_obj_05.json"),
    "tless_toy_05_val": ("tless_toy/obj_05", "tless_toy/obj_05/train_obj_05.json"),
    "tless_pbr_05_train": ("tless_pbr", "tless_pbr/train_obj_05.json"),
    "tless_05_train": ("tless/obj_05", "tless/obj_05/train_obj_05.json"),
    "tless_05_val": ("tless_test", "tless_test/val_obj_05.json"),
    # toy dataset
    "toy_01_train": ("toy/cube", "toy/cube/train_cube.json"),
    "toy_01_val": ("toy/cube", "toy/cube/val_cube.json"),
    "toy_02_train": ("toy/cup", "toy/cup/train_cup.json"),
    "toy_02_val": ("toy/cup", "toy/cup/val_cup.json"),
    "toy_03_train": ("toy/cylinder", "toy/cylinder/train_cylinder.json"),
    "toy_03_val": ("toy/cylinder", "toy/cylinder/val_cylinder.json"),
}

SIXDPOSE_KEYS = ["corner_3d", "corner_2d", "center_3d", "center_2d", "fps_3d", "fps_2d",
                "K", "pose"]

for key, (image_root, json_file) in SPLITS.items():
    # Assume pre-defined datasets live in `./datasets`.
    json_file = os.path.join("datasets", json_file)
    image_root = os.path.join("datasets", image_root)
    meta = get_sixdpose_metadata()

    DatasetCatalog.register(
        key,
        lambda key=key, json_file=json_file, image_root=image_root: load_occlusion_json(
            json_file, image_root, key, extra_annotation_keys=SIXDPOSE_KEYS
        ),
    )

    MetadataCatalog.get(key).set(
        json_file=json_file, image_root=image_root, **meta
    )

if __name__ == "__main__":
    # test load occlusion json
    data_dict = load_occlusion_json(json_file='datasets/occlusion/occlusion_train.json', image_root='datasets/occlusion', dataset_name='occlusion_train')
    print(data_dict[0])