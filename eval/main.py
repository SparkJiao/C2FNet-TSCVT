import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset


def main(cfg):
    # root_dir = cfg.root_dir
    # if cfg.save_dir is not None:
    #     output_dir = cfg.save_dir
    # else:
    #     output_dir = root_dir
    output_dir = cfg.output_dir
    # gt_dir = osp.join(root_dir, 'gt')
    # pred_dir = osp.join(root_dir, 'pred')
    if cfg.methods is None:
        method_names = os.listdir(output_dir)
    else:
        method_names = cfg.methods.split('+')
    if cfg.datasets is None:
        dataset_names = os.listdir(cfg.data_dir)
    else:
        dataset_names = cfg.datasets.split('+')

    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(osp.join(output_dir, method, dataset),
                                 osp.join(cfg.data_dir, dataset, "GT"))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda, cfg.all_metrics)
            threads.append(thread)
    for thread in threads:
        print(thread.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default=None)
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--all_metrics', type=bool, default=True)
    parser.add_argument('--gpu_id', type=str, default='0')
    config = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    main(config)
