#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import numpy as np
import luigi
import z5py

from cluster_tools import MulticutSegmentationWorkflow


def run_wf(sample, tmp_folder, max_jobs, target='local'):

    input_path = '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/validation/predictions/val_block_0%i_unet_lr_v3_bmap.n5' % block_id
    input_key = 'data'
    mask_key = 'data'
    ws_key = ''

    rf_path = ''

    configs = MulticutSegmentationWorkflow.get_config(False)

    if not os.path.exists('config'):
        os.mkdir('config')

    roi_begin, roi_end = None, None

    global_config = configs['global']
    global_config.update({'shebang': "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python",
                          'roi_begin': roi_begin,
                          'roi_end': roi_end})
    with open('./config/global.config', 'w') as f:
        json.dump(global_config, f)

    subprob_config = configs['solve_subproblems']
    subprob_config.update({'weight_edges': True,
                           'threads_per_job': max_jobs})
    with open('./config/solve_subproblems.config', 'w') as f:
        json.dump(subprob_config, f)

    feat_config = configs['block_edge_features']
    if with_rf:
        feat_config.update({'filters': ['gaussianSmoothing'],
                            'sigmas': [(0.5, 1., 1.), (1., 2., 2.),
                                       (2., 4., 4.), (4., 8., 8.)],
                            'halo': (8, 16, 16),
                            'channel_agglomeration': 'mean'})

    else:
        feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]})
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
                                                    mask_path=mask_path, mask_key=mask_key,
                                                    ws_path=input_path, ws_key=ws_key,
                                                    graph_path=exp_path, features_path=exp_path,
                                                    costs_path=exp_path, problem_path=exp_path,
                                                    node_labels_path=exp_path, node_labels_key='node_labels',
                                                    output_path=exp_path, output_key='volumes/segmentation',
                                                    use_decomposition_multicut=False,
                                                    rf_path=rf_path,
                                                    n_scales=2,
                                                    config_dir='./config',
                                                    tmp_folder=tmp_folder,
                                                    target=target,
                                                    max_jobs=max_jobs)], local_scheduler=True)


if __name__ == '__main__':
    with_rf = False

    if with_rf:
        tmp_folder = './tmp_plat_val_rf'
    else:
        tmp_folder = './tmp_plat_val'

    # target = 'slurm'
    # max_jobs = 32

    target = 'local'
    max_jobs = 8

    block_id = 1
    run_wf(block_id, tmp_folder, max_jobs, target=target, with_rf=with_rf)
    # debug_feats()
    # debug_costs()
