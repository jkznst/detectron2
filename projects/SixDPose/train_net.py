# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SixDPose Training Script.

This script is similar to the training script in detectron2/tools.

It is an example of how a user might use detectron2 for a new project.
"""

import os
import torch
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger

# train on linemod
from sixdpose import DatasetMapper, add_sixdpose_config, COCOEvaluator, SixDPoseEvaluator

# train on coco
# from detectron2.evaluation import COCOEvaluator
# from sixdpose import add_sixdpose_config
# from sixdpose import COCODatasetMapper as DatasetMapper

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        if True:
            evaluators.append(SixDPoseEvaluator(dataset_name, cfg, True, output_folder))
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    # def resume_or_load(self, resume=True):
    #     """
    #     If `resume==True`, and last checkpoint exists, resume from it.

    #     Otherwise, load a model specified by the config.

    #     Args:
    #         resume (bool): whether to do resume or not
    #     """
    #     # The checkpoint stores the training iteration that just finished, thus we start
    #     # at the next iteration (or iter zero if there's no checkpoint).
    #     self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)

    #     if resume:
    #         self.start_iter = (self.checkpointer.get("iteration", -1) + 1)
    #     else:
    #         self.start_iter = 0

    def export(self, name=""):
        # export 'model', discard 'optimizer', 'scheduler', 'iteration'
        data = {}
        data["model"] = self.checkpointer.model.state_dict()
        basename = "{}.pth".format(name)
        save_file = os.path.join(self.checkpointer.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        with open(save_file, "wb") as f:
            torch.save(data, f)

    @classmethod
    def vis_feat(cls, cfg, model, outdir='./vis_feat/'):
        model.eval()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            with torch.no_grad():
                for idx, inputs in enumerate(data_loader):
                    print(idx)
                    outputs = model(inputs)
                    # whole img det feat
                    p2=outputs['p2'][0].data.cpu().numpy()
                    p3=outputs['p3'][0].data.cpu().numpy()
                    p4=outputs['p4'][0].data.cpu().numpy()
                    p5=outputs['p5'][0].data.cpu().numpy()

                    # whole img pose feat
                    # p2=outputs[0][0].data.cpu().numpy()
                    # p3=outputs[1][0].data.cpu().numpy()
                    # p4=outputs[2][0].data.cpu().numpy()
                    # p5=outputs[3][0].data.cpu().numpy()
                    # print(p2.shape, p3.shape, p4.shape, p5.shape)
                    np.save(outdir+'{:05d}_p2.npy'.format(idx), p2)
                    np.save(outdir+'{:05d}_p3.npy'.format(idx), p3)
                    np.save(outdir+'{:05d}_p4.npy'.format(idx), p4)
                    np.save(outdir+'{:05d}_p5.npy'.format(idx), p5)

                    # roi_feat = outputs[0].data.cpu().numpy()
                    # np.save(outdir + "{:05d}.npy".format(idx), roi_feat)


def setup(args):
    cfg = get_cfg()
    add_sixdpose_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sixdpose")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    if args.vis_feat:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        Trainer.vis_feat(cfg, model)
        return

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # trainer.export(name="crpnet_Rh_50_FPN_128_dw_res5d_det_1x")
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
