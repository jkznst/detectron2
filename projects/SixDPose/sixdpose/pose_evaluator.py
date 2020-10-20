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

import pycocotools.mask as mask_util
from .pvnet_pose_utils import project, pnp
from .utils.icp import icp_utils
from .utils.img_utils import read_depth
from .utils.vsd import inout
import tqdm

DIAMETERS = {
    # LINEMOD
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
    'phone': 212.358,
    # TLESS
    'obj_05': 108.69,
    # toy
    'obj_01': 138.56410452431246,
    'obj_02': 127.03435730703477,
    'obj_03': 93.20019624216526,
}

def get_ply_model(model_path):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    model = np.stack([x, y, z], axis=-1)
    return model

class SixDPoseLinemodEvaluator(DatasetEvaluator):
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
        # self.model = get_ply_model(model_path) / 1000.
        self.model = inout.load_ply(model_path)
        self.model_pts = self.model['pts'] / 1000.

        self.diameter = DIAMETERS[self.thing_classes] / 1000.

        self.proj2d = []
        self.rot_err = []
        self.t_err = []
        self.add = []
        self.cmd5 = []

        self.icp_add = []
        self.icp_proj2d = []
        self.icp_rot_err = []
        self.icp_t_err = []
        self.icp_cmd5 = []

        self.mask_ap = []

        self.height = 480
        self.width = 640
        self.icp_refiner = icp_utils.ICPRefiner(self.model, (self.width, self.height)) if cfg.TEST.USE_ICP_REFINE else None

    def reset(self):
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.rot_err = []
        self.t_err = []

        self.icp_add = []
        self.icp_proj2d = []
        self.icp_rot_err = []
        self.icp_t_err = []
        self.icp_cmd5 = []

        self.mask_ap = []

    def projection_2d(self, pose_pred, pose_targets, K, icp=False, threshold=5):
        model_2d_pred = project(self.model_pts, K, pose_pred)
        model_2d_targets = project(self.model_pts, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        if icp:
            self.icp_proj2d.append(proj_mean_diff < threshold)
        else:
            self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model_pts, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model_pts, pose_targets[:, :3].T) + pose_targets[:, 3]

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

    def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        t_err = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) / np.linalg.norm(pose_targets[:, 3]) 
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        trace = trace if trace >= -1 else -1
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        if icp:
            self.icp_cmd5.append(translation_distance < 5 and angular_distance < 5)
            self.icp_rot_err.append(angular_distance)
            self.icp_t_err.append(t_err)
        else:
            self.cmd5.append(translation_distance < 5 and angular_distance < 5)
            self.rot_err.append(angular_distance)
            self.t_err.append(t_err)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def icp_refine(self, pose_pred, anno, output, K, img_path):
        img_id = int(img_path.split('/')[-1].strip('.jpg'))
        depth_path = img_path[:-9].replace('JPEGImages', 'depth') + "{:04d}.png".format(img_id)
        depth = read_depth(depth_path)
        mask = mask_util.decode(anno['segmentation'])
        if pose_pred[2, 3] <= 0:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

        R_refined, t_refined = self.icp_refiner.refine(depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(),
                                depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(depth, R_refined, t_refined, K.copy(), no_depth=True)
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
                    if self.icp_refiner is not None:
                        pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K, input['file_name'])
                        if self.thing_classes in ['eggbox', 'glue']:
                            self.add_metric(pose_pred_icp, pose_gt, syn=True, icp=True)
                        else:
                            self.add_metric(pose_pred_icp, pose_gt, icp=True)
                        self.projection_2d(pose_pred_icp, pose_gt, K, icp=True)
                        self.cm_degree_5_metric(pose_pred_icp, pose_gt, icp=True)
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
        print('median translation error metric: {}'.format(t_err))
        # print('mask ap70: {}'.format(ap))
        if self.icp_refiner is not None:
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
            print('2d projections metric after icp: {}'.format(np.mean(self.icp_proj2d)))
            print('5 cm 5 degree metric after icp: {}'.format(np.mean(self.icp_cmd5)))
            print('median rotation error (degree) metric after icp: {}'.format(np.mean(self.icp_rot_err)))
            print('median translation error metric after icp: {}'.format(np.mean(self.icp_t_err)))
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.rot_err = []
        self.t_err = []
        self.mask_ap = []
        self.icp_add = []
        self.icp_proj2d = []
        self.icp_rot_err = []
        self.icp_t_err = []
        self.icp_cmd5 = []
        return {'pose': {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'r': rot_err, 't': t_err}}


class SixDPoseTlessEvaluator(DatasetEvaluator):
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
        # self.model = get_ply_model(model_path) / 1000.
        self.model = inout.load_ply(model_path)
        self.model_pts = self.model['pts'] / 1000.

        self.cfg = cfg

        self.diameter = DIAMETERS[self.thing_classes] / 1000.

        self.rot_err = []
        self.t_err = []
        self.adi = []
        self.cmd5 = []
        self.vsd = []

        self.icp_adi = []
        self.icp_vsd = []
        self.icp_rot_err = []
        self.icp_t_err = []
        self.icp_cmd5 = []

        self.mask_ap = []

        self.pose_per_id = []
        self.pose_icp_per_id = []
        self.img_ids = []

        self.height = 540
        self.width = 720
        self.icp_refiner = icp_utils.ICPRefiner(self.model, (self.width, self.height)) if cfg.TEST.USE_ICP_REFINE or cfg.TEST.VSD else None

    def reset(self):
        self.adi = []
        self.vsd = []
        self.cmd5 = []
        self.rot_err = []
        self.t_err = []

        self.icp_adi = []
        self.icp_vsd = []
        self.icp_rot_err = []
        self.icp_t_err = []
        self.icp_cmd5 = []

        self.mask_ap = []

    def vsd_metric(self, pose_preds, pose_gts, K, depth_path, icp=False):
        from .utils.vsd import vsd_utils

        depth = inout.load_depth(depth_path) * 0.1
        im_size = (depth.shape[1], depth.shape[0])
        dist_test = vsd_utils.misc.depth_im_to_dist_im(depth, K)

        delta = 15
        tau = 20
        cost_type = 'step'
        error_thresh = 0.3

        depth_gt = {}
        dist_gt = {}
        visib_gt = {}

        for pose_pred_ in pose_preds:
            R_est = pose_pred_[:, :3]
            t_est = pose_pred_[:, 3:] * 1000
            depth_est = self.icp_refiner.renderer.render(im_size, 100, 10000, K, R_est, t_est)
            # depth_est = self.opengl.render(im_size, 100, 10000, K, R_est, t_est)
            dist_est = vsd_utils.misc.depth_im_to_dist_im(depth_est, K)

            for gt_id, pose_gt_ in enumerate(pose_gts):
                R_gt = pose_gt_[:, :3]
                t_gt = pose_gt_[:, 3:] * 1000
                if gt_id not in visib_gt:
                    depth_gt_ = self.icp_refiner.renderer.render(im_size, 100, 10000, K, R_gt, t_gt)
                    # depth_gt_ = self.opengl.render(im_size, 100, 10000, K, R_gt, t_gt)
                    dist_gt_ = vsd_utils.misc.depth_im_to_dist_im(depth_gt_, K)
                    dist_gt[gt_id] = dist_gt_
                    visib_gt[gt_id] = vsd_utils.visibility.estimate_visib_mask_gt(
                        dist_test, dist_gt_, delta)

                e = vsd_utils.vsd(dist_est, dist_gt[gt_id], dist_test, visib_gt[gt_id],
                                  delta, tau, cost_type)
                if e < error_thresh:
                    return 1

        return 0

    def adi_metric(self, pose_preds, pose_target, percentage=0.1):
        diameter = self.diameter * percentage
        model_target = np.dot(self.model_pts, pose_target[:, :3].T) + pose_target[:, 3]
        for pose_pred in pose_preds:
            model_pred = np.dot(self.model_pts, pose_pred[:, :3].T) + pose_pred[:, 3]
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_target, k=1)
            mean_dist = np.mean(mean_dist)
            if mean_dist < diameter:
                return 1
        return 0

    def cm_degree_5_metric(self, pose_preds, pose_target):
        for pose_pred in pose_preds:
            translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
            t_err = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) / np.linalg.norm(pose_target[:, 3]) 
            rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
            trace = np.trace(rotation_diff)
            trace = trace if trace <= 3 else 3
            trace = trace if trace >= -1 else -1
            angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
            if translation_distance < 5 and angular_distance < 5:
                return 1
        return 0
        

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def icp_refine(self, pose_pred, anno, output, K, img_path):
        img_id = int(img_path.split('/')[-1].strip('.jpg'))
        depth_path = img_path[:-9].replace('JPEGImages', 'depth') + "{:04d}.png".format(img_id)
        depth = read_depth(depth_path)
        mask = mask_util.decode(anno['segmentation'])
        if pose_pred[2, 3] <= 0:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

        R_refined, t_refined = self.icp_refiner.refine(depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(),
                                depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(depth, R_refined, t_refined, K.copy(), no_depth=True)
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
            instances = output["instances"].to(self._cpu_device)
            # print(instances)
            if self._do_evaluation:
                img_id = int(input['image_id'])
                self.img_ids.append(img_id)
                annos = self._coco_api.loadAnns(self._coco_api.getAnnIds(imgIds=img_id))
                # print(annos[0]['center_2d'])
                if len(annos) > 0:
                    K = np.array(annos[0]['K'])
                    if self._kpt_format == 'fps8':
                        kpt_3d = np.concatenate(([annos[0]['center_3d']], annos[0]['fps_3d']), axis=0)
                    elif self._kpt_format == 'bb8':
                        kpt_3d = np.concatenate(([annos[0]['center_3d']], annos[0]['corner_3d']), axis=0)
                    else:
                        assert self._kpt_format == 'bb8+fps8', self._kpt_format
                        kpt_3d = np.concatenate(([annos[0]['center_3d']], annos[0]['corner_3d'], annos[0]['fps_3d']), axis=0)
                    pose_gts = [np.array(anno['pose']) for anno in annos]
                    
                    pose_preds = []
                    pose_preds_icp = []
                    for i in range(len(instances)):
                        kpt_2d = instances.pred_keypoints[i].numpy()[:, :2]
                        # print(kpt_2d)
                        pose_pred = pnp(kpt_3d, kpt_2d, K)
                        pose_preds.append(pose_pred)
                    self.pose_per_id.append(pose_preds)

                    for pose_gt in pose_gts:
                        self.adi.append(self.adi_metric(pose_preds, pose_gt))
                        self.cmd5.append(self.cm_degree_5_metric(pose_preds, pose_gt))
                    

    def summarize_vsd(self, pose_preds, img_ids, vsd):
        for pose_pred, img_id in tqdm.tqdm(zip(pose_preds, img_ids)):
            img_data = self._coco_api.loadImgs(int(img_id))[0]
            depth_path = os.path.join('/data/ZHANGXIN/DATASETS/SIXD_CHALLENGE/T-LESS/test_primesense_all',
                         '{:06d}'.format((img_id - 1) // 504), 'depth', '{:06d}.png'.format((img_id - 1) % 504))

            ann_ids = self._coco_api.getAnnIds(imgIds=img_id)
            annos = self._coco_api.loadAnns(ann_ids)
            K = np.array(annos[0]['K'])
            pose_gts = [np.array(anno['pose']) for anno in annos]
            
            vsd.append(self.vsd_metric(pose_pred, pose_gts, K, depth_path))

    def evaluate(self):
        print("number of images: {}".format(len(self.img_ids)))
        print("number of annotations: {}".format(len(self.adi)))

        if self.cfg.TEST.VSD:
            from .utils.vsd import vsd_utils
            self.summarize_vsd(self.pose_per_id, self.img_ids, self.vsd)
            if self.cfg.TEST.USE_ICP_REFINE:
                self.summarize_vsd(self.pose_icp_per_id, self.img_ids, self.icp_vsd)
            self.pose_per_id = []
            self.pose_icp_per_id = []
            self.img_ids = []
            
        vsd = np.mean(self.vsd)
        adi = np.mean(self.adi)
        cmd5 = np.mean(self.cmd5)
        # rot_err = np.median(self.rot_err)
        # t_err = np.median(self.t_err)
        # ap = np.mean(self.mask_ap)
        # print('2d projections metric: {}'.format(proj2d))
        print('VSD metric: {}'.format(vsd))
        print('ADI metric: {}'.format(adi))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        # print('median rotation error (degree) metric: {}'.format(rot_err))
        # print('median translation error metric: {}'.format(t_err))
        # print('mask ap70: {}'.format(ap))
        if self.icp_refiner is not None:
            print('ADI metric after icp: {}'.format(np.mean(self.icp_adi)))
            print('5 cm 5 degree metric after icp: {}'.format(np.mean(self.icp_cmd5)))
            # print('median rotation error (degree) metric after icp: {}'.format(np.mean(self.icp_rot_err)))
            # print('median translation error metric after icp: {}'.format(np.mean(self.icp_t_err)))
        # self.proj2d = []
        self.adi = []
        self.cmd5 = []
        # self.rot_err = []
        # self.t_err = []
        # self.mask_ap = []
        self.icp_adi = []
        self.icp_proj2d = []
        self.icp_rot_err = []
        self.icp_t_err = []
        self.icp_cmd5 = []
        return {'pose': {'adi': adi, 'cmd5': cmd5}}