#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi
import h5py
import z5py

from cluster_tools import MulticutSegmentationWorkflow


def run_wf(block_id, tmp_folder, max_jobs,
           target='local'):

    input_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/predictions/val_block_0%i_unet_lr_v3_global_norm.n5' % block_id
    input_key = 'data'
    exp_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/segmentation/val_block_0%i.n5' % block_id

    configs = MulticutSegmentationWorkflow.get_config()

    ws_config = configs['watershed']
    ws_config.update({'threshold': .25, 'apply_presmooth_2d': False,
                      'sigma_weights': (1., 2., 2.), 'apply_dt_2d': False,
                      'pixel_pitch': (2, 1, 1), 'two_pass': True,
                      'halo': [0, 50, 50]})
    with open('./config/watershed.config', 'w') as f:
        json.dump(ws_config ,f)

    feat_config = configs['block_edge_features']
    feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                                    [-2, 0, 0], [0, -4, 0], [0, 0, -4],
                                    [-4, 0, 0], [0, -8, 0], [0, 0, -8]]})
    with open('./config/block_edge_features.config', 'w') as f:
        json.dump(feat_config ,f)

    # set number of threads for sum jobs
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'probs_to_costs',
             'solve_subproblems', 'reduce_problem', 'solve_global']

    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_jobs})
        with open('./config/%s.config' % tt, 'w') as f:
            json.dump(config, f)

    ret = luigi.build([MulticutSegmentationWorkflow(input_path=input_path, input_key=input_key,
                                                    ws_path=exp_path, ws_key='volumes/watershed',
                                                    graph_path=exp_path, features_path=exp_path,
                                                    costs_path=exp_path, problem_path=exp_path,
                                                    node_labels_path=exp_path, node_labels_key='node_labels',
                                                    output_path=exp_path, output_key='volumes/segmentation',
                                                    n_scales=1,
                                                    config_dir='./config',
                                                    tmp_folder=tmp_folder,
                                                    target=target,
                                                    max_jobs=max_jobs)], local_scheduler=True)
    # view the results if we are local and the
    # tasks were successfull
    if ret and target == 'local':
        from cremi_tools.viewer.volumina import view
        with z5py.File(exp_path) as f:
            ds = f['volumes/watershed']
            ds.n_threads = max_jobs
            ws = ds[:]
            ds = f['volumes/segmentation']
            ds.n_threads = max_jobs
            seg = ds[:]

        with z5py.File(input_path) as f:
            ds = f[input_key]
            ds.n_threads = max_jobs
            affs = ds[:]

        view([affs.transpose((1, 2, 3, 0)), ws, seg])



if __name__ == '__main__':
    tmp_folder = '/g/kreshuk/pape/Work/data/cache/plat_val'

    # target = 'slurm'
    # max_jobs = 32

    target = 'local'
    max_jobs = 8

    block_id = 1
    run_wf(block_id, tmp_folder, max_jobs, target=target)
