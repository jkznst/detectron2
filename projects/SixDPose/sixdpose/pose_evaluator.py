# from lib.datasets.dataset_catalog import DatasetCatalog
# from lib.config import cfg
# import pycocotools.coco as coco

# from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
# import os
# from lib.utils.linemod import linemod_config
# import torch
# if cfg.test.icp:
#     from lib.utils import icp_utils
# from PIL import Image
# from lib.utils.img_utils import read_depth
# from scipy import spatial
import contextlib
import io
import json
import logging
import numpy as np
import os
from plyfile import PlyData
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from scipy import spatial

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json

from detectron2.evaluation import DatasetEvaluator

from .pvnet_pose_utils import project, pnp

DIAMETERS = {
    'ape': 102.099,
    'benchvise': 247.506,
    'bowl': 167.355,
    'cam': 172.492,
    'can': 201.404,
    'cat': 154.546,
    'cup': 124.264,
    'driller': 261.472,
    'duck': 108.999,
    'eggbox': 164.628,
    'glue': 175.889,
    'holepuncher': 145.543,
    'iron': 278.078,
    'lamp': 282.601,
    'phone': 212.358
}

def get_ply_model(model_path):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    model = np.stack([x, y, z], axis=-1)
    return model

class SixDPoseEvaluator(DatasetEvaluator):
    '''
    Evaluate object instance sixd pose estimation.
    '''
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_format = cfg.INPUT.KEYPOINT_FORMAT
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

        # pose specific
        self.thing_classes = self._metadata.thing_classes[0]
        model_path = os.path.join(self._metadata.image_root, 'models', self.thing_classes + '.ply')
        self.model = get_ply_model(model_path) / 1000.
        self.diameter = DIAMETERS[self.thing_classes] / 1000.

        self.proj2d = []
        self.rot_err = []
        self.t_err = []
        self.add = []
        self.icp_add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_render = None
        # self.icp_render = icp_utils.SynRenderer(self.thing_classes) if cfg.TEST.USE_ICP_REFINE else None

    def reset(self):
        self.proj2d = []
        self.add = []
        self.icp_add = []
        self.cmd5 = []
        self.rot_err = []
        self.t_err = []
        self.mask_ap = []

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = project(self.model, K, pose_pred)
        model_2d_targets = project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)
        self.rot_err.append(angular_distance)
        self.t_err.append(translation_distance)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0 or np.sum(mask) < 20:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000
        R_refined, t_refined = icp_utils.icp_refinement(depth, self.icp_render, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), (depth.shape[1], depth.shape[0]), depth_only=True,            max_mean_dist_factor=5.0)
        R_refined, _ = icp_utils.icp_refinement(depth, self.icp_render, R_refined, t_refined, K.copy(), (depth.shape[1], depth.shape[0]), no_depth=True)
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))
        return pose_pred

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            instance = output["instances"].to(self._cpu_device)
            if self._do_evaluation:
                img_id = int(input['image_id'])
                anno = self._coco_api.loadAnns(self._coco_api.getAnnIds(imgIds=img_id))
                if len(anno) > 0:
                    anno = anno[0]
                    K = np.array(anno['K'])
                    if self._kpt_format == 'fps8':
                        kpt_3d = np.concatenate(([anno['center_3d']], anno['fps_3d']), axis=0)
                    elif self._kpt_format == 'bb8':
                        kpt_3d = np.concatenate(([anno['center_3d']], anno['corner_3d']), axis=0)
                    else:
                        assert self._kpt_format == 'bb8+fps8', self._kpt_format
                        kpt_3d = np.concatenate(([anno['center_3d']], anno['corner_3d'], anno['fps_3d']), axis=0)

                    if len(instance) > 0:
                        kpt_2d = instance.pred_keypoints[0].numpy()[:, :2]
                    else:
                        kpt_2d = np.zeros((kpt_3d.shape[0], 2))

                    pose_gt = np.array(anno['pose'])
                    pose_pred = pnp(kpt_3d, kpt_2d, K)
                    if self.icp_render is not None:
                        pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K)
                        self.add_metric(pose_pred_icp, pose_gt, icp=True)
                    self.projection_2d(pose_pred, pose_gt, K)
                    if self.thing_classes in ['eggbox', 'glue']:
                        self.add_metric(pose_pred, pose_gt, syn=True)
                    else:
                        self.add_metric(pose_pred, pose_gt)
                    self.cm_degree_5_metric(pose_pred, pose_gt)
                    # self.mask_iou(output, input)

    def evaluate(self):
        print(len(self.proj2d))
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        rot_err = np.median(self.rot_err)
        t_err = np.median(self.t_err)
        # ap = np.mean(self.mask_ap)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('median rotation error (degree) metric: {}'.format(rot_err))
        print('median translation error (cm) metric: {}'.format(t_err))
        # print('mask ap70: {}'.format(ap))
        if self.icp_render is not None:
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.rot_err = []
        self.t_err = []
        self.mask_ap = []
        self.icp_add = []
        return {'pose': {'proj2d': proj2d, 'add': add, 'cmd5': cmd5}}
