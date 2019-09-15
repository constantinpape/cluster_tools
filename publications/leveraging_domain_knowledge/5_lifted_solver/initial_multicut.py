#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
import json
import luigi

from cluster_tools import MulticutSegmentationWorkflow


def initial_mc(max_jobs, max_threads, tmp_folder, target='slurm'):

    n_scales = 1
    input_path = '/g/kreshuk/data/FIB25/cutout.n5'
    exp_path = './exp_data/exp_data.n5'
    input_key = 'volumes/affinities'

    ws_key = 'volumes/segmentation/watershed'
    out_key = 'volumes/segmentation/multicut'
    node_labels_key = 'node_labels/multicut'

    configs = MulticutSegmentationWorkflow.get_config()

    config_folder = './config'
    if not os.path.exists(config_folder):
        os.mkdir(config_folder)

    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python"
    global_config = configs['global']
    global_config.update({'shebang': shebang})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    ws_config = configs['watershed']
    ws_config.update({'threshold': .3, 'apply_presmooth_2d': False,
                      'sigma_weights': 2., 'apply_dt_2d': False,
                      'sigma_seeds': 2., 'apply_ws_2d': False,
                      'two_pass': False, 'alpha': .85,
                      'halo': [25, 25, 25], 'time_limit': 90,
                      'mem_limit': 8, 'size_filter': 100,
                      'channel_begin': 0, 'channel_end': 3})
    with open(os.path.join(config_folder, 'watershed.config'), 'w') as f:
        json.dump(ws_config, f)

    subprob_config = configs['solve_subproblems']
    subprob_config.update({'weight_edges': True,
                           'threads_per_job': max_threads,
                           'time_limit': 180,
                           'mem_limit': 16})
    with open(os.path.join(config_folder, 'solve_subproblems.config'), 'w') as f:
        json.dump(subprob_config, f)

    feat_config = configs['block_edge_features']
    feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                                    [-4, 0, 0], [0, -4, 0], [0, 0, -4]]})
    with open(os.path.join(config_folder, 'block_edge_features.config'), 'w') as f:
        json.dump(feat_config, f)

    # set number of threads for sum jobs
    beta = .5
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'probs_to_costs',
             'reduce_problem']
    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_threads,
                       'mem_limit': 64, 'time_limit': 260,
                       'weight_edges': True, 'beta': beta})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    time_limit_solve = 24*60*60
    config = configs['solve_global']
    config.update({'threads_per_job': max_threads,
                   'mem_limit': 64, 'time_limit': time_limit_solve / 60 + 240,
                   'time_limit_solver': time_limit_solve})
    with open(os.path.join(config_folder, 'solve_global.config'), 'w') as f:
        json.dump(config, f)

    task = MulticutSegmentationWorkflow(input_path=input_path, input_key=input_key,
                                        ws_path=input_path, ws_key=ws_key,
                                        problem_path=exp_path,
                                        node_labels_key=node_labels_key,
                                        output_path=input_path,
                                        output_key=out_key,
                                        n_scales=n_scales,
                                        config_dir=config_folder,
                                        tmp_folder=tmp_folder,
                                        target=target,
                                        max_jobs=max_jobs,
                                        max_jobs_multicut=1,
                                        skip_ws=False)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Multicut segmentation failed"


if __name__ == '__main__':
    # target = 'slurm'
    target = 'local'

    if target == 'slurm':
        max_jobs = 300
        max_threads = 8
    else:
        max_jobs = 64
        max_threads = 16

    tmp_folder = './tmp_mc'
    initial_mc(max_jobs, max_threads, tmp_folder, target=target)
