#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import numpy as np
import luigi
import z5py

from cluster_tools import MulticutSegmentationWorkflow


def run_wf(sample, max_jobs, target='local'):

    tmp_folder = './tmp_%s' % sample
    input_path = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cremi_new/sample%s.n5' % sample
    exp_path = 'sample%s_exp.n5' % sample
    input_key = 'volumes/affinities'
    mask_key = 'masks/original_mask'
    ws_key = 'segmentation/watershed'

    rf_path = ''

    configs = MulticutSegmentationWorkflow.get_config(False)

    if not os.path.exists('config'):
        os.mkdir('config')

    roi_begin, roi_end = None, None

    global_config = configs['global']
    global_config.update({'shebang': "#! /groups/saalfeld/home/papec/Work/software/conda/miniconda3/envs/cluster_env/bin/python",
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
    feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]})
    with open('./config/block_edge_features.config', 'w') as f:
        json.dump(feat_config ,f)

    # set number of threads for sum jobs
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'probs_to_costs',
             'solve_subproblems', 'reduce_problem', 'solve_global']

    for tt in tasks:
        config = configs[tt]
        n_threads = max_jobs if tt != 'reduce_problem' else 4
        config.update({'threads_per_job': n_threads})
        with open('./config/%s.config' % tt, 'w') as f:
            json.dump(config, f)

    ret = luigi.build([MulticutSegmentationWorkflow(input_path=input_path, input_key=input_key,
                                                    mask_path=input_path, mask_key=mask_key,
                                                    ws_path=input_path, ws_key=ws_key,
                                                    graph_path=exp_path, features_path=exp_path,
                                                    costs_path=exp_path, problem_path=exp_path,
                                                    node_labels_path=exp_path, node_labels_key='node_labels',
                                                    output_path=input_path, output_key='segmentation/multicut',
                                                    use_decomposition_multicut=False,
                                                    skip_ws=False,
                                                    rf_path=rf_path,
                                                    n_scales=1,
                                                    config_dir='./config',
                                                    tmp_folder=tmp_folder,
                                                    target=target,
                                                    max_jobs=max_jobs)], local_scheduler=True)
    assert ret, "Sample %s failed" % sample


if __name__ == '__main__':

    target = 'local'
    max_jobs = 32

    samples = ('A',)
    samples = ('A', 'B', 'C', 'A+', 'B+', 'C+')
    for sample in samples:
        run_wf(sample, max_jobs, target=target)
