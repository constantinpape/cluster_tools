#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import numpy as np
import luigi
import z5py

from cluster_tools import MulticutSegmentationWorkflow


def run_mc(sample, tmp_folder, max_jobs, target='local'):

    input_path = '/g/kreshuk/data/cremi/realigned/sample%s_small.n5' % sample
    input_key = 'predictions/full_affs'

    mask_path = ''
    mask_key = ''

    exp_path = 'sample%s_exp.n5' % sample
    rf_path = ''

    use_decomposer = False
    configs = MulticutSegmentationWorkflow.get_config(use_decomposer)

    config_folder = 'config_mc'
    if not os.path.exists(config_folder):
        os.mkdir(config_folder)

    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python"
    global_config = configs['global']
    global_config.update({'shebang': shebang})
    with open('./config_mc/global.config', 'w') as f:
        json.dump(global_config, f)

    subprob_config = configs['solve_subproblems']
    subprob_config.update({'weight_edges': False,
                           'threads_per_job': max_jobs})
    with open('./config_mc/solve_subproblems.config', 'w') as f:
        json.dump(subprob_config, f)

    feat_config = configs['block_edge_features']
    feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]})
    with open('./config_mc/block_edge_features.config', 'w') as f:
        json.dump(feat_config ,f)

    # set number of threads for sum jobs
    if use_decomposer:
        tasks = ['merge_sub_graphs', 'merge_edge_features', 'probs_to_costs',
                 'solve_subproblems', 'decompose', 'insert']
    else:
        tasks = ['merge_sub_graphs', 'merge_edge_features', 'probs_to_costs',
                 'solve_subproblems', 'reduce_problem', 'solve_global']

    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_jobs})
        with open('./config_mc/%s.config' % tt, 'w') as f:
            json.dump(config, f)

    ret = luigi.build([MulticutSegmentationWorkflow(input_path=input_path, input_key=input_key,
                                                    mask_path=mask_path, mask_key=mask_key,
                                                    ws_path=input_path, ws_key='segmentation/watershed',
                                                    graph_path=exp_path, features_path=exp_path,
                                                    costs_path=exp_path, problem_path=exp_path,
                                                    node_labels_path=exp_path, node_labels_key='node_labels',
                                                    output_path=input_path, output_key='segmentation/multicut',
                                                    use_decomposition_multicut=use_decomposer,
                                                    rf_path=rf_path,
                                                    n_scales=1,
                                                    config_dir='./config_mc',
                                                    tmp_folder=tmp_folder,
                                                    target=target,
                                                    skip_ws=True,
                                                    max_jobs=max_jobs)], local_scheduler=True)
    # view the results if we are local and the
    # tasks were successfull
    if ret and target == 'local':
        print("Starting viewer")
        from cremi_tools.viewer.volumina import view

        with z5py.File(input_path) as f:
            ds = f[input_key]
            ds.n_threads = max_jobs
            affs = ds[:]
            if affs.ndim == 4:
                affs = affs.transpose((1, 2, 3, 0))

            ds = f['segmentation/watershed']
            ds.n_threads = max_jobs
            ws = ds[:]

            ds = f['segmentation/multicut']
            ds.n_threads = max_jobs
            seg = ds[:]

        view([affs, ws, seg], ['affs', 'ws', 'mc-seg'])


if __name__ == '__main__':
    sample = 'A'
    tmp_folder = './tmp_mc_%s' % sample

    target = 'local'
    max_jobs = 8

    run_mc(sample, tmp_folder, max_jobs, target=target)
