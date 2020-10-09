# coding:utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def vis_feat():
    pooler_level_file = "pooler_level.txt"
    det_feat_dir = "vis_feat_det"
    pose_feat_dir = "vis_feat_pose"
    vis_diff_dir = "vis_diff"

    pooler_levels = []
    with open(pooler_level_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('[').strip(']\n')
            if len(line) > 0:
                line = line.split(' ')
                line = [int(i) for i in line]
            else:
                line = []
            pooler_levels.append(line)
        f.close()

    for idx, pooler_level in enumerate(pooler_levels):
        if len(pooler_level) == 0:
            continue
        else:
            pooler_level = pooler_level[0]
        det_feat = np.load(os.path.join(det_feat_dir, "{:05d}_p{:01d}.npy".format(idx, pooler_level + 2)))
        pose_feat = np.load(os.path.join(pose_feat_dir, "{:05d}_p{:01d}.npy".format(idx, pooler_level + 2)))
        
        diff = np.abs(det_feat - pose_feat)
        diff = np.sum(diff, axis=0, keepdims=True)
        print(idx)
        for i in range(diff.shape[0]):
            diff_img = diff[i][9:-9, 16:-12]
            diff_img = diff_img / np.max(diff_img)
            # diff_img = sigmoid((diff[i] - np.mean(diff[i])) / np.std(diff[i]) * 2.0)
            diff_img = np.asarray(diff_img * 255, dtype=np.uint8)
            diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
            diff_img = cv2.resize(diff_img, (640, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(vis_diff_dir, "{:05d}.png".format(idx)), diff_img)

def vis_feat_roi():
    det_feat_dir = "vis_feat_det_roi"
    pose_feat_dir = "vis_feat_pose_roi"
    vis_diff_dir = "vis_diff_roi"

    for idx in range(1016):
        det_feat = np.load(os.path.join(det_feat_dir, "{:05d}.npy".format(idx)))
        pose_feat = np.load(os.path.join(pose_feat_dir, "{:05d}.npy".format(idx)))
        
        diff = np.abs(det_feat - pose_feat)
        diff = np.sum(diff, axis=0, keepdims=True)
        print(idx)
        for i in range(diff.shape[0]):
            diff_img = diff[i] / np.max(diff[i])
            diff_img = np.asarray(diff_img * 255, dtype=np.uint8)
            diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
            diff_img = cv2.resize(diff_img, (224, 224), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("win", diff_img)
            # cv2.waitKey()
            cv2.imwrite(os.path.join(vis_diff_dir, "{:05d}.png".format(idx)), diff_img)

def addImage(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    alpha = 0.5
    beta = 1- alpha
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    cv2.imshow("win", img_add)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vis_feat()
    # addImage("/data/ZHANGXIN/pose_estimation_code/SSD-BB8-gluon/data/COCO+LINEMOD_obj05/JPEGImages/00000.jpg", 
    #     "/data/ZHANGXIN/pose_estimation_code/detectron2/vis_diff/00000.png")