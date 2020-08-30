"""
This file is used to generate ../data/COCO+TLESS_fusion_rendering dataset.
Use fusion & rendering strategy
"""
from utils.sixd import load_sixd, load_COCO, load_yaml
from rendering.model import Model3D
from rendering.utils import create_pose, build_6D_poses
from rendering.renderer import Renderer
from tools.fps import fps_utils
import os
from plyfile import PlyData
from lxml.etree import Element, SubElement, tostring
import pycocotools.mask as mask_util
import xml.etree.ElementTree as ET
#import pickle
import matplotlib.pyplot as plt

import random
import math
import cv2
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

palette={1:[128, 0, 0],
        2:[0, 128, 0],
        4:[128, 128, 0],
        5:[0, 0, 128],
        6:[128, 0, 128],
        8:[0, 128, 128],
        9:[128, 128, 128],
        10:[64, 0, 0],
        11:[192, 0, 0],
        12:[64,128,0],
        13:[192,128,0],
        14:[64,0,128],
        15:[192,0,128]}

TLESS_path = '/data/ZHANGXIN/DATASETS/SIXD_CHALLENGE/T-LESS/'
COCO_path = '/data/ZHANGXIN/DATASETS/COCO2017/'
save_path = './data/COCO+TLESS_fusion_rendering/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
scale_to_meters = 0.001

MIN_MARGIN_W = 50
MIN_MARGIN_H = 25

cam_intrinsic = np.identity(3)
cam_info = json.load(os.path.join(TLESS_path, 'tless_base', 'camera_primesense.json'))
cam_intrinsic[0, 0] = cam_info['fx']
cam_intrinsic[0, 2] = cam_info['cx']
cam_intrinsic[1, 1] = cam_info['fy']
cam_intrinsic[1, 2] = cam_info['cy']

window_shape = (540, 720, 3)

CLASSES = ['obj_{:02d}'.format(i) for i in range(1, 31)]    
cat2label = {cat: i + 1 for i, cat in enumerate(CLASSES)}
# cat2cls = {'obj_01': 'ape', 'obj_05': 'can', 'obj_06': 'cat', 'obj_08': 'driller',
#             'obj_09': 'duck', 'obj_10': 'eggbox', 'obj_11': 'glue', 'obj_12': 'holepuncher'}

# gt_info = load_yaml(os.path.join(TLESS_path, 'test', '02', 'gt.yml')) # modify when change model
# occlusion_model_ids = [1, 5, 6, 8, 9, 10, 11, 12]
tless_model_ids = [i for i in range(1, 31)]

views_337 = np.loadtxt('/data/ZHANGXIN/pose_estimation_code/ssd-6d-master/views-337.txt')

rotation_matrix_337 = [create_pose(vertex=views_337[i])[0:3, 0:3] for i in range(len(views_337))]

def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points

def sample_fps_points(model_path, save_path='./fps.txt'):
    ply_path = model_path
    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    np.savetxt(save_path, fps_points)

def cal_intersection(boxA, boxB):
    W = min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])
    H = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])
    if W <= 0 or H <= 0:
        return 0
    else:
        return W * H

def select_closest_view_euclidean_distance(view_point, views):
    least_dist = 100.0
    closest_num = None
    for idx_view in range(len(views)):
        view_dist = np.linalg.norm(view_point - views[idx_view])
        if view_dist < least_dist:
            least_dist = view_dist
            closest_num = idx_view

    return closest_num

def cal_rot_error(rotation1, rotation2):
    return np.arccos((np.trace(np.dot(rotation1.T, rotation2)) - 1) / 2)

def select_closest_view_geometric_distance(view_point, rot_list):
    least_rot_error = 100.0
    closest_num = None

    gt_rotation = create_pose(vertex=view_point)[0:3, 0:3]
    for idx_rot in range(len(rot_list)):
        rot_error = cal_rot_error(gt_rotation, rot_list[idx_rot])
        if rot_error < least_rot_error:
            least_rot_error = rot_error
            closest_num = idx_rot + 1   # view label: 1 ~ 337

    return closest_num

def get_image_paths(folder):
    return (os.path.join(folder, f)
            for f in os.listdir(folder)
            if 'jpeg' in f)

def show_BB8(image, BB8_image_coordinates):
    _, fig = plt.subplots(1, 1, figsize=(6, 6))
    # (3L, 256L, 256L) => (256L, 256L, 3L)
    img = image
    # img = cv2.resize(full_img, dsize=(640, 480))
    img = np.flip(img, axis=2)
    fig.imshow(img)

    rect0 = plt.Line2D([BB8_image_coordinates[0, 0], BB8_image_coordinates[0, 1],
                        BB8_image_coordinates[0, 2], BB8_image_coordinates[0, 3],
                        BB8_image_coordinates[0, 0]],
                       [BB8_image_coordinates[1, 0], BB8_image_coordinates[1, 1],
                        BB8_image_coordinates[1, 2], BB8_image_coordinates[1, 3],
                        BB8_image_coordinates[1, 0]],
                       linewidth=2, color='red')
    rect1 = plt.Line2D([BB8_image_coordinates[0, 4], BB8_image_coordinates[0, 5],
                        BB8_image_coordinates[0, 6], BB8_image_coordinates[0, 7],
                        BB8_image_coordinates[0, 4]],
                       [BB8_image_coordinates[1, 4], BB8_image_coordinates[1, 5],
                        BB8_image_coordinates[1, 6], BB8_image_coordinates[1, 7],
                        BB8_image_coordinates[1, 4]],
                       linewidth=2, color='blue')
    rect2 = plt.Line2D([BB8_image_coordinates[0, 0], BB8_image_coordinates[0, 4]],
                       [BB8_image_coordinates[1, 0], BB8_image_coordinates[1, 4]],
                       linewidth=2, color='green')
    rect3 = plt.Line2D([BB8_image_coordinates[0, 1], BB8_image_coordinates[0, 5]],
                       [BB8_image_coordinates[1, 1], BB8_image_coordinates[1, 5]],
                       linewidth=2, color='yellow')
    rect4 = plt.Line2D([BB8_image_coordinates[0, 2], BB8_image_coordinates[0, 6]],
                       [BB8_image_coordinates[1, 2], BB8_image_coordinates[1, 6]],
                       linewidth=2, color='black')
    rect5 = plt.Line2D([BB8_image_coordinates[0, 3], BB8_image_coordinates[0, 7]],
                       [BB8_image_coordinates[1, 3], BB8_image_coordinates[1, 7]],
                       linewidth=2, color='white')
    fig.add_line(rect0)
    fig.add_line(rect1)
    fig.add_line(rect2)
    fig.add_line(rect3)
    fig.add_line(rect4)
    fig.add_line(rect5)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

def show_fps8(image, fps8_image_coordinates):
    _, fig = plt.subplots(1, 1, figsize=(6, 6))
    # (3L, 256L, 256L) => (256L, 256L, 3L)
    img = image
    # img = cv2.resize(full_img, dsize=(640, 480))
    img = np.flip(img, axis=2)
    fig.imshow(img)

    plt.scatter(fps8_image_coordinates[0, :], fps8_image_coordinates[1, :])
    plt.show()

def draw_random_3d_models(images=[], poses=[], model_ids=[], BoundingBoxes=[]):
    assert len(poses) > 0, "Length of poses is less than 1 !!"
    assert len(images) == len(poses) == len(model_ids) == len(BoundingBoxes), "The lengths of poses and model_ids are different !!"
    w, h = 640, 480

    # augmentation
    with open(os.path.join(COCO_path, 'trainNameList.txt'), 'r') as f:
        coco_train_image_names = f.readlines()
    coco_train_image_names = [name.strip() for name in coco_train_image_names]
    n = np.random.randint(0, len(coco_train_image_names))
    out = cv2.imread(os.path.join(COCO_path, 'train2017', coco_train_image_names[n]))

    out = cv2.resize(out, (640, 480))

    # ground truth: A list of detections for this image, each detection is in the form of
    # [xmin, ymin, xmax, ymax, name, transform (4x4 matrix), view_num, inplane_num]
    detection_list = []
    result_out_labels = []
    ren = Renderer((w, h), cam_intrinsic)
    # draw 3D models in random order
    idx_list = list(range(len(model_ids)))
    random.shuffle(idx_list)
    p = idx_list.index(0)
    # always draw object 02 at last only as an occluder
    idx_list[p], idx_list[-1] = idx_list[-1], idx_list[p]
    images = np.array(images)[idx_list]
    poses = np.array(poses)[idx_list]
    model_ids = np.array(model_ids)[idx_list]
    BoundingBoxes = np.array(BoundingBoxes)[idx_list]

    for image, pose, model_id, BoundingBox in zip(images, poses, model_ids, BoundingBoxes):
        ren.clear()

        model = TLESS_models['obj_{:02d}'.format(model_id)]

        result_out_label = np.zeros(image.shape, dtype=np.int16)

        if pose is not None:
            random_pose = pose
            closest_sampled_inplane = -1
            closest_sampled_view = -1
        else:
            # hemisphere
            longitude_angle = random.uniform(0, 180) / 180 * math.pi
            latitude_angle = random.uniform(0, 180) / 180 * math.pi
            view_point = np.array([math.cos(latitude_angle) * math.cos(longitude_angle),
                                   math.cos(latitude_angle) * math.sin(longitude_angle),
                                   math.sin(latitude_angle)])
            inplane_angle = random.uniform(-45.0, 44.99)
            random_pose = create_pose(vertex=view_point,
                                      angle_deg=inplane_angle)   #random.uniform(-45, 45)
            random_pose[:3, 3] = [0, 0, random.uniform(0.32, 0.48)]  # zr = 0.5
            # determine for the used transformation its
            # closest sampled discrete viewpoint and in-plane rotation (class number)
            closest_sampled_inplane = math.floor((inplane_angle + 45) / 5) + 1  # inplane label:1 ~ 18

            #closest_sampled_view = select_closest_view_euclidean_distance(view_point, views_337)
            closest_sampled_view = select_closest_view_geometric_distance(view_point, rotation_matrix_337)

        ren.draw_model(model, random_pose)
        #ren.draw_boundingbox(model, random_pose)
        col, dep = ren.finish()
        # cv2.imshow('col', col)
        # cv2.waitKey()

        box = np.argwhere(dep)  # Deduct bbox from depth rendering
        # box:[xmin, ymin, xmax, ymax]
        box = [box.min(0)[1], box.min(0)[0], box.max(0)[1] + 1, box.max(0)[0] + 1]
        # box_w, box_h = (box[2] - box[0]), (box[3] - box[1])

        mask = np.dstack((dep, dep, dep)) > 0
        tight_mask = mask[box[1]:box[3], box[0]:box[2]]
        patch = image[box[1]:box[3], box[0]:box[2]]
        resize_ratio = np.random.uniform(0.8, 1.2)
        patch = cv2.resize(patch, (int(patch.shape[1] * resize_ratio), int(patch.shape[0] * resize_ratio)))
        tight_mask = tight_mask.astype(np.float32)
        tight_mask = cv2.resize(tight_mask, (int(tight_mask.shape[1] * resize_ratio),
                                             int(tight_mask.shape[0] * resize_ratio)))
        tight_mask = tight_mask.astype(np.bool)
        box_w = patch.shape[1]
        box_h = patch.shape[0]

        out_mask = np.zeros(out.shape, dtype=bool)
        # pixel_label = 10 * (idx_model + 1) * np.ones(shape=tight_mask.shape, dtype=np.int16)
        # todo: make seg mask
        pixel_label = palette[model_id] * np.ones(shape=tight_mask.shape, dtype=np.uint8)

        patience = 10
        for i in range(patience):
            # random_position:[xmin, ymin]
            random_position = np.array([0, 0], dtype=np.int16)
            if (out.shape[1] - box_w < 0) or (out.shape[0] - box_h < 0):
                break

            # augmentation
            margin_w = np.int(np.minimum(MIN_MARGIN_W, (out.shape[1] - box_w) / 2 - 1))
            margin_h = np.int(np.minimum(MIN_MARGIN_H, (out.shape[0] - box_h) / 2 - 1))
            random_position[0] = np.random.randint(margin_w, out.shape[1] - box_w - margin_w)
            random_position[1] = np.random.randint(margin_h, out.shape[0] - box_h - margin_h)

            # judge occlusion by bounding box
            over_occlusion = False
            for idx_dec in range(len(detection_list)):
                existing_box = detection_list[idx_dec][:4]
                new_box = [random_position[0], random_position[1], random_position[0]+box_w, random_position[1]+box_h]
                intersection = cal_intersection(existing_box, new_box)
                area_of_existing_box = (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1])
                area_of_new_box = box_w * box_h
                if (intersection / area_of_new_box > 0.75) or (intersection / area_of_existing_box > 0.75):
                    over_occlusion = True
                    break
            # existing_label = result_out_label[random_position[1]:random_position[1] + box_h,
            #                  random_position[0]:random_position[0] + box_w].astype(bool)
            # Intersection = existing_label & tight_mask
            # Union = existing_label | tight_mask
            # number_of_intersection = len(Intersection[Intersection>0])
            # number_of_union = len(Union[Union>0])
            # IoU = number_of_intersection / number_of_union

            if over_occlusion:
                continue
            else:
                out_mask[random_position[1]:random_position[1] + box_h, random_position[0]:random_position[0] + box_w] \
                    = tight_mask
                #out[out_mask] = col[mask]
                #out[out_mask] = image[mask] * 255
                out[out_mask] = patch[tight_mask] * 255
                # cv2.imshow('image', out)
                # cv2.waitKey()
                result_out_label[out_mask] = pixel_label[tight_mask]
                # detection: [xmin, ymin, xmax, ymax, name, pose(4x4 matrix),
                # closest_sampled_view, closest_sampled_inplane]
                detection = [random_position[0], random_position[1], random_position[0]+box_w, random_position[1]+box_h]
                #detection.append('obj_{:02d}'.format(idx_model+1))
                detection.append('obj_{:02d}'.format(model_id))
                detection.append(random_pose)
                detection.append(closest_sampled_view)
                detection.append(closest_sampled_inplane)

                origin_obj = np.zeros(shape=(4,1))
                origin_obj[3,0] = 1
                origin_cam = np.dot(random_pose, origin_obj)
                origin_cam = origin_cam[0:3] / origin_cam[2]
                origin_img = np.dot(cam_intrinsic, origin_cam)

                center = [random_position[0] - resize_ratio * (box[0] - origin_img[0,0]),   # x
                          random_position[1] - resize_ratio * (box[1] - origin_img[1,0])]   # y
                #out[int(center[1]), int(center[0]), :] = np.array([0, 255, 0])
                detection.append(center)

                # calculate image coordinates of 8 boundingbox corners
                transform = random_pose
                transform = np.reshape(transform, newshape=(4, 4))
                transform = np.round(transform, 4)

                BoundingBox_image_coordinates = np.dot(transform, BoundingBox)
                BoundingBox_image_coordinates = BoundingBox_image_coordinates[0:3, :]
                BoundingBox_image_coordinates = BoundingBox_image_coordinates / BoundingBox_image_coordinates[2, :]
                BoundingBox_image_coordinates = np.dot(cam_intrinsic, BoundingBox_image_coordinates)
                BoundingBox_image_coordinates[0] = random_position[0] + resize_ratio * (BoundingBox_image_coordinates[0] - box[0])
                BoundingBox_image_coordinates[1] = random_position[1] + resize_ratio * (BoundingBox_image_coordinates[1] - box[1])
                show_BB8(out, BoundingBox_image_coordinates)

                BoundingBox_image_coordinates = BoundingBox_image_coordinates[0:2]
                BoundingBox_image_coordinates[0] = BoundingBox_image_coordinates[0] / window_shape[1]   # x
                BoundingBox_image_coordinates[1] = BoundingBox_image_coordinates[1] / window_shape[0]   # y
                BoundingBox_image_coordinates = np.round(BoundingBox_image_coordinates, 4)
                detection.append(BoundingBox_image_coordinates)

                detection_list.append(detection)
                result_out_labels.append(result_out_label)
                break


    return out, result_out_labels, detection_list

def synthesize_by_fusion_multi(images=[], poses=[], model_ids=[]):
    assert len(poses) > 0, "Length of poses is less than 1 !!"
    assert len(images) == len(poses) == len(model_ids), "The lengths of poses and model_ids are different !!"
    w, h = 640, 480

    # augmentation
    with open(os.path.join(COCO_path, 'trainNameList.txt'), 'r') as f:
        coco_train_image_names = f.readlines()
    coco_train_image_names = [name.strip() for name in coco_train_image_names]
    n = np.random.randint(0, len(coco_train_image_names))
    out = cv2.imread(os.path.join(COCO_path, 'train2017', coco_train_image_names[n]))
    out = cv2.resize(out, (w, h))

    seg_appearance_label = np.zeros(out.shape, dtype=np.uint8)
    ins_appearance_label = np.zeros(out.shape, dtype=np.uint8)
    ins_appearance_label_list = []
    ins_occ_agnostic_label_list = []
    valid_ins_idxs = []

    # ground truth: A list of detections for this image, each detection is in the form of
    # [xmin, ymin, xmax, ymax, name, transform (4x4 matrix), view_num, inplane_num]
    detection_list = []
    ren = Renderer((w, h), cam_intrinsic)

    # number_of_instances = len(poses)
    ins_color_interval = 10

    # draw 3D models in random order
    idx_list = list(range(len(model_ids)))
    random.shuffle(idx_list)
    p = idx_list.index(0)
    # always draw object 02 at last only as an occluder
    idx_list[p], idx_list[-1] = idx_list[-1], idx_list[p]
    images = np.array(images)[idx_list]
    poses = np.array(poses)[idx_list]
    model_ids = np.array(model_ids)[idx_list]

    for ins_idx, (image, pose, model_id) in enumerate(zip(images, poses, model_ids)):

        ren.clear()

        model = TLESS_models['obj_{:02d}'.format(model_id)]

        if pose is not None:
            random_pose = pose
            closest_sampled_inplane = -1
            closest_sampled_view = -1
        else:
            # hemisphere
            longitude_angle = random.uniform(0, 180) / 180 * math.pi
            latitude_angle = random.uniform(0, 180) / 180 * math.pi
            view_point = np.array([math.cos(latitude_angle) * math.cos(longitude_angle),
                                   math.cos(latitude_angle) * math.sin(longitude_angle),
                                   math.sin(latitude_angle)])
            inplane_angle = random.uniform(-45.0, 44.99)
            random_pose = create_pose(vertex=view_point,
                                      angle_deg=inplane_angle)   #random.uniform(-45, 45)
            random_pose[:3, 3] = [0, 0, random.uniform(0.32, 0.48)]  # zr = 0.5
            # determine for the used transformation its
            # closest sampled discrete viewpoint and in-plane rotation (class number)
            closest_sampled_inplane = math.floor((inplane_angle + 45) / 5) + 1  # inplane label:1 ~ 18

            #closest_sampled_view = select_closest_view_euclidean_distance(view_point, views_337)
            closest_sampled_view = select_closest_view_geometric_distance(view_point, rotation_matrix_337)

        ren.draw_model(model, random_pose)
        #ren.draw_boundingbox(model, random_pose)
        col, dep = ren.finish()
        # cv2.imshow('col', col)
        # cv2.waitKey()

        box = np.argwhere(dep)  # Deduct bbox from depth rendering
        # box:[xmin, ymin, xmax, ymax]
        box = [box.min(0)[1], box.min(0)[0], box.max(0)[1] + 1, box.max(0)[0] + 1]
        # box_w, box_h = (box[2] - box[0]), (box[3] - box[1])

        mask = np.dstack((dep, dep, dep)) > 0
        tight_mask = mask[box[1]:box[3], box[0]:box[2]]
        patch = image[box[1]:box[3], box[0]:box[2]]
        resize_ratio = np.random.uniform(0.8, 1.2)
        patch = cv2.resize(patch, (int(patch.shape[1] * resize_ratio), int(patch.shape[0] * resize_ratio)))
        tight_mask = tight_mask.astype(np.float32)
        tight_mask = cv2.resize(tight_mask, (int(tight_mask.shape[1] * resize_ratio),
                                             int(tight_mask.shape[0] * resize_ratio)))
        tight_mask = tight_mask.astype(np.bool)
        box_w = patch.shape[1]
        box_h = patch.shape[0]

        out_mask = np.zeros(out.shape, dtype=bool)
        # pixel_label = 10 * (idx_model + 1) * np.ones(shape=tight_mask.shape, dtype=np.int16)
        # todo: make seg mask
        if model_id in occlusion_model_ids:
            seg_pixel_label = palette[model_id] * np.ones(shape=tight_mask.shape, dtype=np.uint8)
            ins_pixel_label = (ins_idx + 1) * ins_color_interval * np.ones(shape=tight_mask.shape, dtype=np.uint8)
        else:
            seg_pixel_label = np.zeros(shape=tight_mask.shape, dtype=np.uint8)
            ins_pixel_label = np.zeros(shape=tight_mask.shape, dtype=np.uint8)

        patience = 10
        for i in range(patience):
            # random_position:[xmin, ymin]
            random_position = np.array([0, 0], dtype=np.int16)
            if (out.shape[1] - box_w < 0) or (out.shape[0] - box_h < 0):
                break

            # augmentation
            margin_w = np.int(np.minimum(MIN_MARGIN_W, (out.shape[1] - box_w) / 2 - 1))
            margin_h = np.int(np.minimum(MIN_MARGIN_H, (out.shape[0] - box_h) / 2 - 1))
            random_position[0] = np.random.randint(margin_w, out.shape[1] - box_w - margin_w)
            random_position[1] = np.random.randint(margin_h, out.shape[0] - box_h - margin_h)

            # judge occlusion by bounding box
            over_occlusion = False
            for idx_dec in range(len(detection_list)):
                existing_box = detection_list[idx_dec][:4]
                new_box = [random_position[0], random_position[1], random_position[0]+box_w, random_position[1]+box_h]
                intersection = cal_intersection(existing_box, new_box)
                area_of_existing_box = (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1])
                area_of_new_box = box_w * box_h
                if (intersection / area_of_new_box > 0.75) or (intersection / area_of_existing_box > 0.75):
                    over_occlusion = True
                    break
            # existing_label = result_out_label[random_position[1]:random_position[1] + box_h,
            #                  random_position[0]:random_position[0] + box_w].astype(bool)
            # Intersection = existing_label & tight_mask
            # Union = existing_label | tight_mask
            # number_of_intersection = len(Intersection[Intersection>0])
            # number_of_union = len(Union[Union>0])
            # IoU = number_of_intersection / number_of_union

            if over_occlusion:
                continue
            else:
                out_mask[random_position[1]:random_position[1] + box_h, random_position[0]:random_position[0] + box_w] \
                    = tight_mask
                #out[out_mask] = col[mask]
                #out[out_mask] = image[mask] * 255
                out[out_mask] = patch[tight_mask] * 255
                # cv2.imshow('image', out)
                # cv2.waitKey()
                seg_appearance_label[out_mask] = seg_pixel_label[tight_mask]
                ins_appearance_label[out_mask] = ins_pixel_label[tight_mask]

                if model_id in occlusion_model_ids:
                    valid_ins_idxs.append(ins_idx)
                    ins_occ_agnostic_label_list.append(out_mask.astype(np.float32) * 255.0)
                    # detection: [xmin, ymin, xmax, ymax, name, pose(4x4 matrix),
                    # closest_sampled_view, closest_sampled_inplane]
                    detection = [random_position[0], random_position[1], random_position[0]+box_w, random_position[1]+box_h]
                    #detection.append('obj_{:02d}'.format(idx_model+1))
                    detection.append('obj_{:02d}'.format(model_id))
                    detection.append(random_pose)
                    detection.append(closest_sampled_view)
                    detection.append(closest_sampled_inplane)

                    origin_obj = np.zeros(shape=(4,1))
                    origin_obj[3,0] = 1
                    origin_cam = np.dot(random_pose, origin_obj)
                    origin_cam = origin_cam[0:3] / origin_cam[2]
                    origin_img = np.dot(cam_intrinsic, origin_cam)

                    center = [random_position[0] - resize_ratio * (box[0] - origin_img[0,0]),   # x
                            random_position[1] - resize_ratio * (box[1] - origin_img[1,0])]   # y
                    #out[int(center[1]), int(center[0]), :] = np.array([0, 255, 0])
                    center[0] = center[0] / window_shape[1]
                    center[1] = center[1] / window_shape[0]
                    detection.append(center)

                    # calculate image coordinates of 8 boundingbox corners
                    transform = random_pose
                    transform = np.reshape(transform, newshape=(4, 4))
                    transform = np.round(transform, 4)

                    BoundingBox_image_coordinates = np.dot(transform, model.bb8)
                    BoundingBox_image_coordinates = BoundingBox_image_coordinates[0:3, :]
                    BoundingBox_image_coordinates = BoundingBox_image_coordinates / BoundingBox_image_coordinates[2, :]
                    BoundingBox_image_coordinates = np.dot(cam_intrinsic, BoundingBox_image_coordinates)
                    BoundingBox_image_coordinates[0] = random_position[0] + resize_ratio * (BoundingBox_image_coordinates[0] - box[0])
                    BoundingBox_image_coordinates[1] = random_position[1] + resize_ratio * (BoundingBox_image_coordinates[1] - box[1])
                    # show_BB8(out, BoundingBox_image_coordinates)

                    BoundingBox_image_coordinates = BoundingBox_image_coordinates[0:2]
                    BoundingBox_image_coordinates[0] = BoundingBox_image_coordinates[0] / window_shape[1]   # x
                    BoundingBox_image_coordinates[1] = BoundingBox_image_coordinates[1] / window_shape[0]   # y
                    BoundingBox_image_coordinates = np.round(BoundingBox_image_coordinates, 4)
                    detection.append(BoundingBox_image_coordinates)

                    fps8_image_coordinates = np.dot(transform, model.fps8)
                    fps8_image_coordinates = fps8_image_coordinates[0:3, :]
                    fps8_image_coordinates = fps8_image_coordinates / fps8_image_coordinates[2, :]
                    fps8_image_coordinates = np.dot(cam_intrinsic, fps8_image_coordinates)
                    fps8_image_coordinates[0] = random_position[0] + resize_ratio * (fps8_image_coordinates[0] - box[0])
                    fps8_image_coordinates[1] = random_position[1] + resize_ratio * (fps8_image_coordinates[1] - box[1])
                    # show_fps8(out, fps8_image_coordinates)

                    fps8_image_coordinates = fps8_image_coordinates[0:2]
                    fps8_image_coordinates[0] = fps8_image_coordinates[0] / window_shape[1]   # x
                    fps8_image_coordinates[1] = fps8_image_coordinates[1] / window_shape[0]   # y
                    fps8_image_coordinates = np.round(fps8_image_coordinates, 4)
                    # print(fps8_image_coordinates)
                    detection.append(fps8_image_coordinates)

                    detection_list.append(detection)
                break

    # cv2.imshow('image', ins_appearance_label)
    # cv2.waitKey()   
    for valid_ins_idx in valid_ins_idxs:
        ins_appearance_label_list.append((ins_appearance_label[:,:,0:1] == ((valid_ins_idx + 1) * ins_color_interval)).astype(np.float32) * 255.0)
        # cv2.imshow('image', (ins_appearance_label[:,:,0:1] == ((ins_idx + 1) * ins_color_interval)).astype(np.float32) * 255.0)
        # cv2.waitKey()

    return out, seg_appearance_label, ins_appearance_label_list, ins_occ_agnostic_label_list, detection_list

def synthesize_by_rendering(poses=[], model_ids=[]):
    assert len(poses) > 0, "Length of poses is less than 1 !!"
    assert len(poses) == len(model_ids), "The lengths of poses and model_ids are different !!"
    w, h = 640, 480

    # augmentation
    with open(os.path.join(COCO_path, 'trainNameList.txt'), 'r') as f:
        coco_train_image_names = f.readlines()
    coco_train_image_names = [name.strip() for name in coco_train_image_names]
    n = np.random.randint(0, len(coco_train_image_names))
    out = cv2.imread(os.path.join(COCO_path, 'train2017', coco_train_image_names[n]))
    out = cv2.resize(out, (w, h))

    seg_appearance_label = np.zeros(out.shape, dtype=np.uint8)
    ins_appearance_label = np.zeros(out.shape, dtype=np.uint8)
    ins_appearance_label_list = []
    ins_occ_agnostic_label_list = []
    valid_ins_idxs = []

    # ground truth: A list of detections for this image, each detection is in the form of
    # [xmin, ymin, xmax, ymax, name, transform (4x4 matrix), view_num, inplane_num]
    detection_list = []
    ren = Renderer((w, h), cam_intrinsic)

    # number_of_instances = len(poses)
    ins_color_interval = 10

    # draw 3D models in random order
    idx_list = list(range(len(model_ids)))
    random.shuffle(idx_list)
    
    poses = np.array(poses)[idx_list]
    model_ids = np.array(model_ids)[idx_list]

    for ins_idx, (pose, model_id) in enumerate(zip(poses, model_ids)):
        ren.clear()

        model = TLESS_models['obj_{:02d}'.format(model_id)]

        if pose is not None:
            random_pose = pose
            closest_sampled_inplane = -1
            closest_sampled_view = -1
        else:
            # hemisphere
            longitude_angle = random.uniform(0, 180) / 180 * math.pi
            latitude_angle = random.uniform(0, 180) / 180 * math.pi
            view_point = np.array([math.cos(latitude_angle) * math.cos(longitude_angle),
                                   math.cos(latitude_angle) * math.sin(longitude_angle),
                                   math.sin(latitude_angle)])
            inplane_angle = random.uniform(-45.0, 44.99)
            random_pose = create_pose(vertex=view_point,
                                      angle_deg=inplane_angle)   #random.uniform(-45, 45)
            random_pose[:3, 3] = [0, 0, random.uniform(0.8, 1.2)]  # zr = 0.5
            # determine for the used transformation its
            # closest sampled discrete viewpoint and in-plane rotation (class number)
            closest_sampled_inplane = math.floor((inplane_angle + 45) / 5) + 1  # inplane label:1 ~ 18

            #closest_sampled_view = select_closest_view_euclidean_distance(view_point, views_337)
            closest_sampled_view = select_closest_view_geometric_distance(view_point, rotation_matrix_337)

        ren.draw_model(model, random_pose)
        #ren.draw_boundingbox(model, random_pose)
        col, dep = ren.finish()
        # cv2.imshow('col', col)
        # cv2.waitKey()

        box = np.argwhere(dep)  # Deduct bbox from depth rendering
        # box:[xmin, ymin, xmax, ymax]
        box = [box.min(0)[1], box.min(0)[0], box.max(0)[1] + 1, box.max(0)[0] + 1]
        # box_w, box_h = (box[2] - box[0]), (box[3] - box[1])

        mask = np.dstack((dep, dep, dep)) > 0
        tight_mask = mask[box[1]:box[3], box[0]:box[2]]
        patch = col[box[1]:box[3], box[0]:box[2]]
        resize_ratio = np.random.uniform(0.8, 1.2)
        patch = cv2.resize(patch, (int(patch.shape[1] * resize_ratio), int(patch.shape[0] * resize_ratio)))
        tight_mask = tight_mask.astype(np.float32)
        tight_mask = cv2.resize(tight_mask, (int(tight_mask.shape[1] * resize_ratio),
                                             int(tight_mask.shape[0] * resize_ratio)))
        tight_mask = tight_mask.astype(np.bool)
        box_w = patch.shape[1]
        box_h = patch.shape[0]

        out_mask = np.zeros(out.shape, dtype=bool)
        # pixel_label = 10 * (idx_model + 1) * np.ones(shape=tight_mask.shape, dtype=np.int16)
        # todo: make seg mask
        if model_id in occlusion_model_ids:
            seg_pixel_label = palette[model_id] * np.ones(shape=tight_mask.shape, dtype=np.uint8)
            ins_pixel_label = (ins_idx + 1) * ins_color_interval * np.ones(shape=tight_mask.shape, dtype=np.uint8)
        else:
            seg_pixel_label = np.zeros(shape=tight_mask.shape, dtype=np.uint8)
            ins_pixel_label = np.zeros(shape=tight_mask.shape, dtype=np.uint8)

        patience = 10
        for i in range(patience):
            # random_position:[xmin, ymin]
            random_position = np.array([0, 0], dtype=np.int16)
            if (out.shape[1] - box_w < 0) or (out.shape[0] - box_h < 0):
                break

            # augmentation
            margin_w = np.int(np.minimum(MIN_MARGIN_W, (out.shape[1] - box_w) / 2 - 1))
            margin_h = np.int(np.minimum(MIN_MARGIN_H, (out.shape[0] - box_h) / 2 - 1))
            random_position[0] = np.random.randint(margin_w, out.shape[1] - box_w - margin_w)
            random_position[1] = np.random.randint(margin_h, out.shape[0] - box_h - margin_h)

            # judge occlusion by bounding box
            over_occlusion = False
            for idx_dec in range(len(detection_list)):
                existing_box = detection_list[idx_dec][:4]
                new_box = [random_position[0], random_position[1], random_position[0]+box_w, random_position[1]+box_h]
                intersection = cal_intersection(existing_box, new_box)
                area_of_existing_box = (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1])
                area_of_new_box = box_w * box_h
                if (intersection / area_of_new_box > 0.75) or (intersection / area_of_existing_box > 0.75):
                    over_occlusion = True
                    break
            # existing_label = result_out_label[random_position[1]:random_position[1] + box_h,
            #                  random_position[0]:random_position[0] + box_w].astype(bool)
            # Intersection = existing_label & tight_mask
            # Union = existing_label | tight_mask
            # number_of_intersection = len(Intersection[Intersection>0])
            # number_of_union = len(Union[Union>0])
            # IoU = number_of_intersection / number_of_union

            if over_occlusion:
                continue
            else:
                out_mask[random_position[1]:random_position[1] + box_h, random_position[0]:random_position[0] + box_w] \
                    = tight_mask
                #out[out_mask] = col[mask]
                #out[out_mask] = image[mask] * 255
                out[out_mask] = patch[tight_mask] * 255
                # cv2.imshow('image', out)
                # cv2.waitKey()
                seg_appearance_label[out_mask] = seg_pixel_label[tight_mask]
                ins_appearance_label[out_mask] = ins_pixel_label[tight_mask]

                if model_id in occlusion_model_ids:
                    valid_ins_idxs.append(ins_idx)
                    ins_occ_agnostic_label_list.append(out_mask.astype(np.float32) * 255.0)
                    # detection: [xmin, ymin, xmax, ymax, name, pose(4x4 matrix),
                    # closest_sampled_view, closest_sampled_inplane]
                    detection = [random_position[0], random_position[1], random_position[0]+box_w, random_position[1]+box_h]
                    #detection.append('obj_{:02d}'.format(idx_model+1))
                    detection.append('obj_{:02d}'.format(model_id))
                    detection.append(random_pose)
                    detection.append(closest_sampled_view)
                    detection.append(closest_sampled_inplane)

                    origin_obj = np.zeros(shape=(4,1))
                    origin_obj[3,0] = 1
                    origin_cam = np.dot(random_pose, origin_obj)
                    origin_cam = origin_cam[0:3] / origin_cam[2]
                    origin_img = np.dot(cam_intrinsic, origin_cam)

                    center = [random_position[0] - resize_ratio * (box[0] - origin_img[0,0]),   # x
                            random_position[1] - resize_ratio * (box[1] - origin_img[1,0])]   # y
                    # out[int(center[1]), int(center[0]), :] = np.array([0, 255, 0])
                    center[0] = center[0] / window_shape[1]
                    center[1] = center[1] / window_shape[0]
                    detection.append(center)

                    # calculate image coordinates of 8 boundingbox corners
                    transform = random_pose
                    transform = np.reshape(transform, newshape=(4, 4))
                    transform = np.round(transform, 4)

                    BoundingBox_image_coordinates = np.dot(transform, model.bb8)
                    BoundingBox_image_coordinates = BoundingBox_image_coordinates[0:3, :]
                    BoundingBox_image_coordinates = BoundingBox_image_coordinates / BoundingBox_image_coordinates[2, :]
                    BoundingBox_image_coordinates = np.dot(cam_intrinsic, BoundingBox_image_coordinates)
                    BoundingBox_image_coordinates[0] = random_position[0] + resize_ratio * (BoundingBox_image_coordinates[0] - box[0])
                    BoundingBox_image_coordinates[1] = random_position[1] + resize_ratio * (BoundingBox_image_coordinates[1] - box[1])
                    # show_BB8(out, BoundingBox_image_coordinates)

                    BoundingBox_image_coordinates = BoundingBox_image_coordinates[0:2]
                    BoundingBox_image_coordinates[0] = BoundingBox_image_coordinates[0] / window_shape[1]   # x
                    BoundingBox_image_coordinates[1] = BoundingBox_image_coordinates[1] / window_shape[0]   # y
                    BoundingBox_image_coordinates = np.round(BoundingBox_image_coordinates, 4)
                    detection.append(BoundingBox_image_coordinates)

                    fps8_image_coordinates = np.dot(transform, model.fps8)
                    fps8_image_coordinates = fps8_image_coordinates[0:3, :]
                    fps8_image_coordinates = fps8_image_coordinates / fps8_image_coordinates[2, :]
                    fps8_image_coordinates = np.dot(cam_intrinsic, fps8_image_coordinates)
                    fps8_image_coordinates[0] = random_position[0] + resize_ratio * (fps8_image_coordinates[0] - box[0])
                    fps8_image_coordinates[1] = random_position[1] + resize_ratio * (fps8_image_coordinates[1] - box[1])
                    # show_fps8(out, fps8_image_coordinates)

                    fps8_image_coordinates = fps8_image_coordinates[0:2]
                    fps8_image_coordinates[0] = fps8_image_coordinates[0] / window_shape[1]   # x
                    fps8_image_coordinates[1] = fps8_image_coordinates[1] / window_shape[0]   # y
                    fps8_image_coordinates = np.round(fps8_image_coordinates, 4)
                    # print(fps8_image_coordinates)
                    detection.append(fps8_image_coordinates)

                    detection_list.append(detection)
                break

    # cv2.imshow('image', ins_appearance_label)
    # cv2.waitKey()   
    for valid_ins_idx in valid_ins_idxs:
        ins_appearance_label_list.append((ins_appearance_label[:,:,0:1] == ((valid_ins_idx + 1) * ins_color_interval)).astype(np.float32) * 255.0)
        # cv2.imshow('image', (ins_appearance_label[:,:,0:1] == ((ins_idx + 1) * ins_color_interval)).astype(np.float32) * 255.0)
        # cv2.waitKey()

    return out, seg_appearance_label, ins_appearance_label_list, ins_occ_agnostic_label_list, detection_list

def draw_original_3d_models(image, poses=[], model_ids=[]):
    assert len(poses) > 0, "Length of poses is less than 1 !!"
    assert len(poses) == len(model_ids), "The lengths of poses and model_ids are different !!"

    w, h = 640, 480

    # original
    out = np.copy(image)
    out = cv2.resize(out, (w, h))
    seg_appearance_label = np.zeros(image.shape, dtype=np.uint8)
    ins_appearance_label = np.zeros(out.shape, dtype=np.uint8)
    ins_appearance_label_list = []
    ins_occ_agnostic_label_list = []

    # ground truth: A list of detections for this image, each detection is in the form of
    # [xmin, ymin, xmax, ymax, name, transform (4x4 matrix), view_num, inplane_num]
    detection_list = []
    ren = Renderer((w, h), cam_intrinsic)

    number_of_instances = len(poses)
    ins_color_interval = 10

    for ins_idx, (model_id, pose) in enumerate(zip(model_ids, poses)):

        ren.clear()

        model = TLESS_models['obj_{:02d}'.format(model_id)]

        random_pose = pose
        closest_sampled_inplane = -1
        closest_sampled_view = -1

        ren.draw_model(model, random_pose)
        #ren.draw_boundingbox(model, random_pose)
        col, dep = ren.finish()
        # cv2.imshow('col', col)
        # cv2.waitKey()

        box = np.argwhere(dep)  # Deduct bbox from depth rendering
        # box:[xmin, ymin, xmax, ymax]
        if box.size > 0:
            box = [box.min(0)[1], box.min(0)[0], box.max(0)[1] + 1, box.max(0)[0] + 1]
        else:
            continue

        mask = np.dstack((dep, dep, dep)) > 0
        tight_mask = mask[box[1]:box[3], box[0]:box[2]]
        patch = image[box[1]:box[3], box[0]:box[2]]
        resize_ratio = 1. #np.random.uniform(0.8, 1.2)
        patch = cv2.resize(patch, (int(patch.shape[1] * resize_ratio), int(patch.shape[0] * resize_ratio)))
        tight_mask = tight_mask.astype(np.float32)
        tight_mask = cv2.resize(tight_mask, (int(tight_mask.shape[1] * resize_ratio),
                                             int(tight_mask.shape[0] * resize_ratio)))
        tight_mask = tight_mask.astype(np.bool)
        box_w = patch.shape[1]
        box_h = patch.shape[0]

        out_mask = np.zeros(out.shape, dtype=bool)
        # pixel_label = 10 * (idx_model + 1) * np.ones(shape=tight_mask.shape, dtype=np.int16)
        seg_pixel_label = palette[model_id] * np.ones(shape=tight_mask.shape, dtype=np.uint8)
        ins_pixel_label = (ins_idx + 1) * ins_color_interval * np.ones(shape=tight_mask.shape, dtype=np.uint8)

        # random_position:[xmin, ymin]
        random_position = np.array([0, 0], dtype=np.int16)
        # original
        random_position[0] = box[0]
        random_position[1] = box[1]


        out_mask[random_position[1]:random_position[1] + box_h, random_position[0]:random_position[0] + box_w] \
                = tight_mask
        ins_occ_agnostic_label_list.append(out_mask.astype(np.float32) * 255.0)
        #out[out_mask] = col[mask]
        #out[out_mask] = image[mask] * 255
        out[out_mask] = patch[tight_mask]
        # cv2.imshow('image', out)
        # cv2.waitKey()
        seg_appearance_label[out_mask] = seg_pixel_label[tight_mask]
        ins_appearance_label[out_mask] = ins_pixel_label[tight_mask]
        # detection: [xmin, ymin, xmax, ymax, name, pose(4x4 matrix),
        # closest_sampled_view, closest_sampled_inplane]
        detection = [random_position[0], random_position[1], random_position[0]+box_w, random_position[1]+box_h]
        #detection.append('obj_{:02d}'.format(idx_model+1))
        detection.append('obj_{:02d}'.format(model_id))
        detection.append(random_pose)
        detection.append(closest_sampled_view)
        detection.append(closest_sampled_inplane)

        origin_obj = np.zeros(shape=(4,1))
        origin_obj[3,0] = 1
        origin_cam = np.dot(random_pose, origin_obj)
        origin_cam = origin_cam[0:3] / origin_cam[2]
        origin_img = np.dot(cam_intrinsic, origin_cam)

        center = [random_position[0] - resize_ratio * (box[0] - origin_img[0,0]),   # x
                        random_position[1] - resize_ratio * (box[1] - origin_img[1,0])]   # y
        # out[int(center[1]), int(center[0]), :] = np.array([0, 255, 0])
        center[0] = center[0] / window_shape[1]
        center[1] = center[1] / window_shape[0]
        detection.append(center)

        # calculate image coordinates of 8 boundingbox corners
        transform = random_pose
        transform = np.reshape(transform, newshape=(4, 4))
        transform = np.round(transform, 4)

        BoundingBox_image_coordinates = np.dot(transform, model.bb8)
        BoundingBox_image_coordinates = BoundingBox_image_coordinates[0:3, :]
        BoundingBox_image_coordinates = BoundingBox_image_coordinates / BoundingBox_image_coordinates[2, :]
        BoundingBox_image_coordinates = np.dot(cam_intrinsic, BoundingBox_image_coordinates)
        # show_BB8(out, BoundingBox_image_coordinates)

        BoundingBox_image_coordinates = BoundingBox_image_coordinates[0:2]
        BoundingBox_image_coordinates[0] = BoundingBox_image_coordinates[0] / window_shape[1]   # x
        BoundingBox_image_coordinates[1] = BoundingBox_image_coordinates[1] / window_shape[0]   # y
        BoundingBox_image_coordinates = np.round(BoundingBox_image_coordinates, 4)
        detection.append(BoundingBox_image_coordinates)

        fps8_image_coordinates = np.dot(transform, model.fps8)
        fps8_image_coordinates = fps8_image_coordinates[0:3, :]
        fps8_image_coordinates = fps8_image_coordinates / fps8_image_coordinates[2, :]
        fps8_image_coordinates = np.dot(cam_intrinsic, fps8_image_coordinates)
        # show_fps8(out, fps8_image_coordinates)

        fps8_image_coordinates = fps8_image_coordinates[0:2]
        fps8_image_coordinates[0] = fps8_image_coordinates[0] / window_shape[1]   # x
        fps8_image_coordinates[1] = fps8_image_coordinates[1] / window_shape[0]   # y
        fps8_image_coordinates = np.round(fps8_image_coordinates, 4)
        # print(fps8_image_coordinates)
        detection.append(fps8_image_coordinates)

        detection_list.append(detection)
    
    # cv2.imshow('image', ins_appearance_label)
    # cv2.waitKey()   
    for ins_idx in range(number_of_instances):
        ins_appearance_label_list.append((ins_appearance_label[:,:,0:1] == ((ins_idx + 1) * ins_color_interval)).astype(np.float32) * 255.0)
        # cv2.imshow('image', (ins_appearance_label[:,:,0:1] == ((ins_idx + 1) * ins_color_interval)).astype(np.float32) * 255.0)
        # cv2.waitKey()

    return out, seg_appearance_label, ins_appearance_label_list, ins_occ_agnostic_label_list, detection_list

def save_xml(info, out_file_path):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'COCO+OCCLUSION'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(info[0])

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(info[2])

    node_height = SubElement(node_size, 'height')
    node_height.text = str(info[1])

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '1'

    for i in range(len(info[3])):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(info[3][i][4])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(info[3][i][0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(info[3][i][1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(info[3][i][2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(info[3][i][3])
        node_transform = SubElement(node_object, 'transform')
        node_transform.text = ','.join(str(k) for k in info[3][i][5].reshape(16))
        node_view = SubElement(node_object, 'view')
        node_view.text = str(info[3][i][6])
        node_inplane = SubElement(node_object, 'inplane')
        node_inplane.text = str(info[3][i][7])
        node_center = SubElement(node_object, 'center')
        node_center.text = ','.join(str(i) for i in info[3][i][8])
        node_BB8 = SubElement(node_object, 'BB8')
        node_BB8.text = ','.join(str(k) for k in info[3][i][9].T.reshape(16))
        node_fps8 = SubElement(node_object, 'fps8')
        node_fps8.text = ','.join(str(k) for k in info[3][i][10].T.reshape(16))

    xml = tostring(node_root, pretty_print=True, encoding='unicode')
    # xml_file_dirname = os.path.dirname(out_file_path)
    # if not os.path.exists(xml_file_dirname):
    #    os.mkdir(xml_file_dirname)
    if not os.path.exists(os.path.dirname(out_file_path)):
        os.mkdir(os.path.dirname(out_file_path))
    fh = open(out_file_path, 'w')
    fh.write(xml)
    fh.close()

def record_occ_ann(model_meta, anno_file, data_root, img_id, ann_id, images, annotations, cls_names=[]):
    print("recording {} information in occlusion dataset ...".format(', '.join(str(cn) for cn in cls_names)))
    K = cam_intrinsic

    img_type = 'real'
    if 'train' in anno_file:
        img_type = 'synthetic'

    inds = np.loadtxt(anno_file, np.str)
    inds = [int(ind) for ind in inds]

    # rgb_dir = os.path.join(data_root, 'JPEGImages')
    xml_dir = os.path.join(os.path.dirname(anno_file).replace('ImageSets/Main', ''), 'Annotations')
    for ind in tqdm(inds):
        img_name = '{:05d}.jpg'.format(ind)
        # rgb_path = os.path.join(rgb_dir, img_name)

        xml_name = '{:05d}.xml'.format(ind)
        xml_path = os.path.join(xml_dir, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        img_id += 1
        info = {'file_name': 'JPEGImages/' + img_name, 'height': height, 'width': width, 'id': img_id, 
                'seg_map': 'SegmentationClass/' + img_name.replace('.jpg', '.png')}
        images.append(info)

        for ins_idx, obj in enumerate(root.findall('object')):
            name = obj.find('name').text
            if name in cls_names:
                # label = cat2label[name] # for multi classes
                label = 1   # for single class
                cls = cat2cls[name]
                bnd_box = obj.find('bndbox')
                bbox = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]  # coco format (xmin, ymin, w, h)

                transform = obj.find('transform').text
                transform = [float(trans) for trans in transform.split(',')]
                transform = np.array(transform).reshape((4, 4))
                # print(transform)
                pose = transform[0:3, :]    # shape (3, 4)

                # center_2d = project(center_3d[None], K, pose)[0]
                corner_3d = model_meta[name].bb8.T[:, 0:3]  # shape (8, 3)
                corner_2d = obj.find('BB8').text
                corner_2d = [float(i) for i in corner_2d.split(',')]
                corner_2d = np.array(corner_2d).reshape((-1, 2)) * np.array([width, height], dtype=np.float)  # x,y

                center_3d = np.zeros((3,), dtype=np.float)
                center_2d = obj.find('center').text
                center_2d = [float(i) for i in center_2d.split(',')]
                center_2d = np.array(center_2d) * np.array([width, height], dtype=np.float)  # x,y

                fps_3d = model_meta[name].fps8.T[:, 0:3]    # shape (8, 3)
                fps_2d = obj.find('fps8').text
                fps_2d = [float(i) for i in fps_2d.split(',')]
                fps_2d = np.array(fps_2d).reshape((-1, 2)) * np.array([width, height], dtype=np.float)  # x,y

                mask_path = os.path.join(data_root, 'SegmentationObject', '{:05d}_{:03d}.png'.format(ind, ins_idx))
                mask_occ_agn_path = os.path.join(data_root, 'SegmentationObjectOccAgn', '{:05d}_{:03d}.png'.format(ind, ins_idx))

                if not os.path.exists(mask_path):
                    print("instance with no appearance mask !!!!")
                    continue
                mask = cv2.imread(mask_path)
                # cv2.imshow("1", mask)
                # cv2.waitKey(0)
                mask = np.asfortranarray(mask)
                mask = mask_util.encode(mask)[0]
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                mask["counts"] = mask["counts"].decode("utf-8")
                # print(mask['counts'])

                # regen_mask = mask_util.decode(mask) * 255
                # cv2.imshow("1", regen_mask)
                # cv2.waitKey(0)

                if not os.path.exists(mask_occ_agn_path):
                    print("instance with no occlusion-agnostic mask !!!!")
                    continue
                mask_occ_agn = cv2.imread(mask_occ_agn_path)
                # cv2.imshow("1", mask)
                # cv2.waitKey(0)
                mask_occ_agn = np.asfortranarray(mask_occ_agn)
                mask_occ_agn = mask_util.encode(mask_occ_agn)[0]
                mask_occ_agn_area = mask_util.area(mask_occ_agn)
                # print(mask_occ_agn_area)
                mask_occ_agn["counts"] = mask_occ_agn["counts"].decode("utf-8")

                # # regen_mask_occ_agn = mask_util.decode(mask_occ_agn) * 255
                # # cv2.imshow("1", regen_mask)
                # # cv2.waitKey(0)

                ann_id += 1
                anno = {'segmentation': mask, 'segmentation_occagn': mask_occ_agn, 'bbox': bbox,
                        'image_id': img_id, 'category_id': label, 'id': ann_id, 'area': int(mask_occ_agn_area)}
                anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
                anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
                anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
                anno.update({'K': K.tolist(), 'pose': pose.tolist()})
                # anno.update({'data_root': rgb_dir})
                anno.update({'type': img_type, 'cls': cls})
                annotations.append(anno)

    return img_id, ann_id

def occlusion_to_coco(anno_file, data_root, model_meta, cls_name=None):
    img_id = 0
    ann_id = 0
    images = []
    annotations = []
    categories = []

    if cls_name is None:
        cls_name_list = CLASSES
        categories = [{'supercategory': 'none', 'id': 1, 'name': 'ape'},
        {'supercategory': 'none', 'id': 2, 'name': 'can'},
        {'supercategory': 'none', 'id': 3, 'name': 'cat'},
        {'supercategory': 'none', 'id': 4, 'name': 'driller'},
        {'supercategory': 'none', 'id': 5, 'name': 'duck'},
        {'supercategory': 'none', 'id': 6, 'name': 'eggbox'},
        {'supercategory': 'none', 'id': 7, 'name': 'glue'},
        {'supercategory': 'none', 'id': 8, 'name': 'holepuncher'}]
    elif isinstance(cls_name, str):
        cls_name_list = [cls_name]
        categories.append({'supercategory': 'none', 'id': 1, 'name': cat2cls[cls_name]})
    else:
        assert isinstance(cls_name, list), "cls_name should be list[str] or None !!"
        cls_name_list = cls_name
        for i, cn in enumerate(cls_name):
            categories.append({'supercategory': 'none', 'id': i + 1, 'name': cat2cls[cn]})

    img_id, ann_id = record_occ_ann(model_meta, anno_file, data_root, img_id, ann_id, images, annotations, cls_names=cls_name_list)
    print('total images = ', img_id)
    print('total annos = ', ann_id)

    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    if cls_name is None:
        json_anno_path = anno_file.replace('.txt', '.json')
    elif isinstance(cls_name, str):
        json_anno_path = anno_file.replace('.txt', '_{}.json'.format(cat2cls[cls_name]))
    else:
        assert isinstance(cls_name, list), "cls_name should be list[str] or None !!"
        json_anno_path = anno_file.replace('.txt', '_{}.json'.format('_'.join(cat2cls[cn] for cn in cls_name)))

    print("Saving annotations to {}.".format(json_anno_path))
    with open(json_anno_path, 'w') as f:
        json.dump(instance, f)

if __name__ == "__main__":
    
    orig_model_ids = [2]

    TLESS_models = {}  # dict
    print('Loading 3D models from TLESS dataset...')
    models_info = json.load(os.path.join(TLESS_path, 'tless_models','models_cad', 'models_info.json'))
    for key, val in models_info.items():
        if key in tless_model_ids:
            name = 'obj_{:02d}'.format(int(key))
            TLESS_models[name] = Model3D()
            TLESS_models[name].load(os.path.join(TLESS_path, 'tless_models/models_cad/' + name + '.ply'), scale=scale_to_meters)
            TLESS_models[name].diameter = val['diameter']
            if not os.path.exists(os.path.join(TLESS_path, 'models', 'fps8_{:02d}.txt'.format(int(key)))):
                sample_fps_points(model_path=os.path.join(TLESS_path, 'models/' + name + '.ply'), save_path=os.path.join(TLESS_path, 'models', 'fps8_{:02d}.txt'.format(int(key))))
            TLESS_models[name].fps8 = np.loadtxt(os.path.join(TLESS_path, 'models', 'fps8_{:02d}.txt'.format(int(key))))
            TLESS_models[name].fps8 *= scale_to_meters
            TLESS_models[name].fps8 = np.insert(TLESS_models[name].fps8, 3, np.ones(8, dtype=np.float), axis=1)
            TLESS_models[name].fps8 = TLESS_models[name].fps8.T
            # print(TLESS_models[name].fps8)

            model_objxx_info = models_info[key]
            min_x = model_objxx_info['min_x'] * scale_to_meters
            min_y = model_objxx_info['min_y'] * scale_to_meters
            min_z = model_objxx_info['min_z'] * scale_to_meters
            size_x = model_objxx_info['size_x'] * scale_to_meters
            size_y = model_objxx_info['size_y'] * scale_to_meters
            size_z = model_objxx_info['size_z'] * scale_to_meters
            max_x = min_x + size_x
            max_y = min_y + size_y
            max_z = min_z + size_z

            BoundingBox = np.zeros(shape=(8, 4))
            BoundingBox[0, :] = np.array([min_x, min_y, min_z, 1.0])
            BoundingBox[1, :] = np.array([min_x, min_y, max_z, 1.0])
            BoundingBox[2, :] = np.array([min_x, max_y, max_z, 1.0])
            BoundingBox[3, :] = np.array([min_x, max_y, min_z, 1.0])
            BoundingBox[4, :] = np.array([max_x, min_y, min_z, 1.0])
            BoundingBox[5, :] = np.array([max_x, min_y, max_z, 1.0])
            BoundingBox[6, :] = np.array([max_x, max_y, max_z, 1.0])
            BoundingBox[7, :] = np.array([max_x, max_y, min_z, 1.0])
            TLESS_models[name].bb8 = BoundingBox.T  # 4 x 8

    imageset_path = os.path.join(TLESS_path, 'test', '{:02}'.format(2), 'rgb')

    image_name_file = open(os.path.join(TLESS_path, 'test', '{:02}'.format(2), 'image_names.txt'), 'r')
    image_path_list = image_name_file.readlines()
    image_name_file.close()
    image_path_list = [os.path.join(imageset_path, f.strip('\n')) for f in image_path_list]
    real_image_num = len(image_path_list)

    JPEGImage_path = save_path + 'JPEGImages/'
    if not os.path.exists(JPEGImage_path):
        os.mkdir(JPEGImage_path)
    Segment_path = save_path + 'SegmentationClass/'
    if not os.path.exists(Segment_path):
        os.mkdir(Segment_path)
    InstanceApp_path = save_path + 'SegmentationObject/'
    if not os.path.exists(InstanceApp_path):
        os.mkdir(InstanceApp_path)
    InstanceOccAgn_path = save_path + 'SegmentationObjectOccAgn/'
    if not os.path.exists(InstanceOccAgn_path):
        os.mkdir(InstanceOccAgn_path)
    ImageSets_path = save_path + 'ImageSets/Main/'
    if not os.path.exists(ImageSets_path):
        os.makedirs(ImageSets_path)

    # transform pose annotations for real images
    print('Transforming pose annotations for real images...')
    print("total number of real images = {:d}".format(real_image_num))
    # for idx_image in tqdm(range(real_image_num)):
    #     image = cv2.imread(image_path_list[idx_image]).astype(np.float32) / 255.0
    #     resize_img = cv2.resize(src=image, dsize=(640, 480))
    #     # cv2.imshow('1', resize_img)
    #     # cv2.waitKey()
    
    #     current_gt_poses = []
    #     current_model_ids = []
    #     for idx_instance in range(len(gt_info[idx_image])):
    #         cid = gt_info[idx_image][idx_instance]['obj_id']
    #         if cid in occlusion_model_ids:
    #             gt_rotation = np.array(gt_info[idx_image][idx_instance]['cam_R_m2c'])
    #             gt_translation = np.array(gt_info[idx_image][idx_instance]['cam_t_m2c']) * scale_to_meters
    
    #             temp_gt_pose = np.eye(4)
    #             temp_gt_pose[0:3, 0:3] = gt_rotation.reshape((3,3))
    #             temp_gt_pose[0:3, 3:4] = gt_translation.reshape((3,1))
    
    #             current_gt_poses.append(temp_gt_pose)
    #             current_model_ids.append(cid)
    
    #     out, seg_app_label, ins_app_label_list, ins_occ_agn_label_list, gt_list = draw_original_3d_models(resize_img, poses=current_gt_poses, model_ids=current_model_ids)
    #     # cv2.imshow('1', out)
    #     # cv2.imshow('label', out_label / 255.0)
    #     # cv2.waitKey()
    
    #     # save transformed pose annotations
    #     cv2.imwrite(JPEGImage_path + '{:05d}.jpg'.format(idx_image), out * 255)
    #     cv2.imwrite(Segment_path + '{:05d}.png'.format(idx_image), seg_app_label)
    #     for ins_idx, (ins_app_label, ins_occ_agn_label) in enumerate(zip(ins_app_label_list, ins_occ_agn_label_list)):
    #             cv2.imwrite(InstanceApp_path + '{:05d}_{:03d}.png'.format(idx_image, ins_idx), ins_app_label)
    #             cv2.imwrite(InstanceOccAgn_path + '{:05d}_{:03d}.png'.format(idx_image, ins_idx), ins_occ_agn_label)
    
    #     # [image_name, image_height, image_width, detection_list]
    #     xml_gt_info = ['{:05d}.jpg'.format(idx_image), resize_img.shape[0], resize_img.shape[1], gt_list]
    #     save_xml(xml_gt_info, save_path + 'Annotations/' + '{:05d}.xml'.format(idx_image))

    # write val.txt
    # with open(ImageSets_path + 'val.txt', 'w') as f:
    #     for i in range(real_image_num):
    #         f.write('{:05d}\n'.format(i))
    #     f.close()

    # # convert to coco json format
    # occlusion_to_coco(anno_file=os.path.join(ImageSets_path, 'val.txt'), 
    #                 data_root='data/COCO+OCCLUSION_fusion_rendering', model_meta=TLESS_models, cls_name='obj_12')

    # synthesize training set by fusion, extract object patches from corresponding sequences
    print('synthesize training set by fusion...')
    fusion_image_num = 10000
    print('total number of fusion images = {:d}'.format(fusion_image_num))
    # for idx in tqdm(range(0, fusion_image_num)):   # if resume, modify here
    #     # idx_image = train_sets[2][idx]
    #     # image = cv2.imread(seq02_train_image_names[idx]).astype(np.float32) / 255.0
    #     idx_image = np.random.randint(0, real_image_num)
    #     image = cv2.imread(image_path_list[idx_image]).astype(np.float32) / 255.0
    #     seq02_resize_img = cv2.resize(src=image, dsize=(640, 480))

    #     #find gt infomation of object02 in seq02
    #     current_source_images = []
    #     current_gt_poses = []
    #     current_model_ids = []

    #     # randomly extract object02 patches
    #     for idx_instance in range(len(gt_info[idx_image])):
    #         cid = gt_info[idx_image][idx_instance]['obj_id']
    #         if cid in orig_model_ids:
    #             gt_rotation = np.array(gt_info[idx_image][idx_instance]['cam_R_m2c'])
    #             gt_translation = np.array(gt_info[idx_image][idx_instance]['cam_t_m2c']) * scale_to_meters

    #             seq02_gt_pose = np.eye(4)
    #             seq02_gt_pose[0:3, 0:3] = gt_rotation.reshape((3, 3))
    #             seq02_gt_pose[0:3, 3:4] = gt_translation.reshape((3, 1))

    #             current_source_images.append(seq02_resize_img)
    #             current_gt_poses.append(seq02_gt_pose)
    #             current_model_ids.append(cid)

    #     # ***********************************fusion augmentation*************************************
    #     for model_id in tless_model_ids:
    #         if model_id in orig_model_ids:
    #             pass

    #         else:
    #             temp_gt_info = load_yaml(os.path.join(TLESS_path, 'test', '{:02d}'.format(model_id), 'gt.yml'))
    #             rand_idx = np.random.randint(0, len(temp_gt_info))

    #             image = cv2.imread(os.path.join(TLESS_path, 'test', '{:02}'.format(model_id), 'rgb',
    #                                             '{:04d}.png'.format(rand_idx))).astype(np.float32) / 255.0
    #             resize_img = cv2.resize(src=image, dsize=(640, 480))

    #             gt_rotation = np.array(temp_gt_info[rand_idx][0]['cam_R_m2c'])
    #             gt_translation = np.array(temp_gt_info[rand_idx][0]['cam_t_m2c']) * scale_to_meters

    #             temp_gt_pose = np.eye(4)
    #             temp_gt_pose[0:3, 0:3] = gt_rotation.reshape((3, 3))
    #             temp_gt_pose[0:3, 3:4] = gt_translation.reshape((3, 1))

    #             current_source_images.append(resize_img)
    #             current_gt_poses.append(temp_gt_pose)
    #             current_model_ids.append(model_id)

    #     out, seg_app_label, ins_app_label_list, ins_occ_agn_label_list, gt_list = synthesize_by_fusion_multi(current_source_images, poses=current_gt_poses,
    #                                                                                 model_ids=current_model_ids)
    #     # cv2.imshow('1', out)
    #     # cv2.imshow('label', out_label / 255.0)
    #     # cv2.waitKey()

    #     # save synthetic training images
    #     cv2.imwrite(JPEGImage_path + '{:05d}.jpg'.format(idx + real_image_num), out)
    #     cv2.imwrite(Segment_path + '{:05d}.png'.format(idx + real_image_num), seg_app_label)
    #     for ins_idx, (ins_app_label, ins_occ_agn_label) in enumerate(zip(ins_app_label_list, ins_occ_agn_label_list)):
    #             cv2.imwrite(InstanceApp_path + '{:05d}_{:03d}.png'.format(idx + real_image_num, ins_idx), ins_app_label)
    #             cv2.imwrite(InstanceOccAgn_path + '{:05d}_{:03d}.png'.format(idx + real_image_num, ins_idx), ins_occ_agn_label)

    #     # [image_name, image_height, image_width, detection_list]
    #     xml_gt_info = ['{:05d}.jpg'.format(idx + real_image_num), resize_img.shape[0], resize_img.shape[1], gt_list]
    #     save_xml(xml_gt_info, save_path + 'Annotations/' + '{:05d}.xml'.format(idx + real_image_num))

    #     # display
    #     # cv2.imshow('1', out)
    #     # cv2.imshow('label', out_label / 255.0)
    #     # cv2.waitKey()

    # synthesize training set by rendering
    rendering_image_num = 10000
    print('total number of rendering images = {:d}'.format(rendering_image_num))
    # for idx in tqdm(range(rendering_image_num)):
    #     current_gt_poses = []
    #     current_model_ids = []
        
    #     for model_id in tless_model_ids:
    #         current_gt_poses.append(None)
    #         current_model_ids.append(model_id)

    #     out, seg_app_label, ins_app_label_list, ins_occ_agn_label_list, gt_list = synthesize_by_rendering(current_gt_poses, current_model_ids)
    #     # cv2.imshow('1', out)
    #     # cv2.imshow('label', out_label / 255.0)
    #     # cv2.waitKey()

    #     # save synthetic training images
    #     cv2.imwrite(JPEGImage_path + '{:05d}.jpg'.format(idx + fusion_image_num + real_image_num), out)
    #     cv2.imwrite(Segment_path + '{:05d}.png'.format(idx + fusion_image_num + real_image_num), seg_app_label)
    #     for ins_idx, (ins_app_label, ins_occ_agn_label) in enumerate(zip(ins_app_label_list, ins_occ_agn_label_list)):
    #         cv2.imwrite(InstanceApp_path + '{:05d}_{:03d}.png'.format(idx + fusion_image_num + real_image_num, ins_idx), ins_app_label)
    #         cv2.imwrite(InstanceOccAgn_path + '{:05d}_{:03d}.png'.format(idx + fusion_image_num + real_image_num, ins_idx), ins_occ_agn_label)

    #     # [image_name, image_height, image_width, detection_list]
    #     xml_gt_info = ['{:05d}.jpg'.format(idx + fusion_image_num + real_image_num), out.shape[0], out.shape[1], gt_list]
    #     save_xml(xml_gt_info, save_path + 'Annotations/' + '{:05d}.xml'.format(idx + fusion_image_num + real_image_num))
    
    # write train.txt
    # with open(ImageSets_path + 'train.txt', 'w') as f:
    #     for i in range(real_image_num, real_image_num + fusion_image_num + rendering_image_num):
    #         f.write('{:05d}\n'.format(i))
    #     f.close()

    # convert to coco json format
    occlusion_to_coco(anno_file=os.path.join(ImageSets_path, 'train.txt'), 
                    data_root='data/COCO+OCCLUSION_fusion_rendering', model_meta=TLESS_models, cls_name='obj_11')