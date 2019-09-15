#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
import json
import luigi

from cluster_tools import MulticutSegmentationWorkflow
from cluster_tools import LiftedMulticutSegmentationWorkflow


def run_mc(target, max_jobs, max_threads):
    input_path = '/g/kreshuk/data/arendt/sponge/data.n5'
    input_key = 'volumes/predictions/nn/affs'

    exp_path = '/g/kreshuk/data/arendt/sponge/exp_data/mc.n5'
    ws_key = 'volumes/segmentation/nn/watershed'
    seg_key = 'volumes/segmentation/nn/multicut'
    assignment_key = 'node_labels/nn/multicut'

    config_folder = './configs'
    configs = MulticutSegmentationWorkflow.get_config()

    graph_config = configs['initial_sub_graphs']
    graph_config.update({'qos': 'normal',  'mem_limit': 4})

    ws_config = configs['watershed']
    ws_config.update({'threshold': .25, 'apply_dt_2d': False, 'apply_ws_2d': False,
                      'size_filter': 100, 'alpha': .9, 'non_maximum_suppression': True,
                      'mem_limit': 8})
    with open(os.path.join(config_folder, 'watershed.config'), 'w') as f:
        json.dump(ws_config, f)

    subprob_config = configs['solve_subproblems']
    subprob_config.update({'threads_per_job': max_threads,
                           'time_limit': 720,
                           'mem_limit': 64,
                           'qos': 'normal',
                           'time_limit_solver': 60*60*6})
    with open(os.path.join(config_folder, 'solve_subproblems.config'), 'w') as f:
        json.dump(subprob_config, f)

    feat_config = configs['block_edge_features']
    feat_config.update({'offsets': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]})
    with open(os.path.join(config_folder, 'block_edge_features.config'), 'w') as f:
        json.dump(feat_config, f)

    weight_edges = True
    exponent = 1.
    costs_config = configs['probs_to_costs']
    costs_config.update({'weight_edges': weight_edges,
                         'weighting_exponent': exponent,
                         'mem_limit': 16, 'qos': 'normal',
                         'beta': 0.5})
    with open(os.path.join(config_folder, 'probs_to_costs.config'), 'w') as f:
        json.dump(costs_config, f)

    # set number of threads for sum jobs
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'map_edge_ids',
             'reduce_problem', 'solve_global']
    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_threads if tt != 'reduce_problem' else 8,
                       'mem_limit': 128,
                       'time_limit': 1440,
                       'qos': 'normal',
                       'agglomerator': 'decomposition-gaec',
                       'time_limit_solver': 60*60*15})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    n_scales = 1
    max_jobs_mc = 6
    tmp_folder = './tmp_mc'
    task = MulticutSegmentationWorkflow(input_path=input_path, input_key=input_key,
                                        ws_path=input_path, ws_key=ws_key,
                                        problem_path=exp_path,
                                        node_labels_key=assignment_key,
                                        output_path=input_path,
                                        output_key=seg_key,
                                        n_scales=n_scales,
                                        config_dir=config_folder,
                                        tmp_folder=tmp_folder,
                                        target=target,
                                        max_jobs=max_jobs,
                                        max_jobs_multicut=max_jobs_mc,
                                        sanity_checks=False,
                                        skip_ws=False)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Multicut failed"


def run_lmc(target, max_jobs, max_threads, mode):

    input_path = '/g/kreshuk/data/arendt/sponge/data.n5'
    input_key = 'volumes/predictions/boundaries'
    ws_key = 'volumes/segmentation/watershed_new'

    if mode == 'same':
        exp_path = '/g/kreshuk/data/arendt/sponge/exp_data/lifted_multicut.n5'
        seg_key = 'volumes/segmentation/for_eval/lifted_multicut'
        assignment_key = 'node_labels/for_eval/lifted_multicut'
        tmp_folder = './tmp_lmc'
    elif mode == 'all':
        exp_path = '/g/kreshuk/data/arendt/sponge/exp_data/lifted_multicut_all.n5'
        seg_key = 'volumes/segmentation/for_eval/lifted_multicut_all'
        assignment_key = 'node_labels/for_eval/lifted_multicut_all'
        tmp_folder = './tmp_lmc_all'
    else:
        raise ValueError(mode)

    config_folder = './configs'
    configs = LiftedMulticutSegmentationWorkflow.get_config()

    subprob_config = configs['solve_lifted_subproblems']
    subprob_config.update({'threads_per_job': max_threads,
                           'time_limit': 720,
                           'mem_limit': 384,
                           'qos': 'normal',
                           'time_limit_solver': 60*60*18})
    with open(os.path.join(config_folder, 'solve_lifted_subproblems.config'), 'w') as f:
        json.dump(subprob_config, f)

    # set number of threads for sum jobs
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'map_edge_ids',
             'reduce_lifted_problem', 'solve_lifted_global']

    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_threads,
                       'mem_limit': 256,
                       'time_limit': 1440,
                       'qos': 'normal',
                       'time_limit_solver': 60*60*15})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    tasks = ['block_node_labels', 'merge_node_labels']
    for tt in tasks:
        config = configs[tt]
        config.update({"time_limit": 160, "mem_limit": 16})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    conf = configs['sparse_lifted_neighborhood']
    conf.update({'time_limit': 240, 'mem_limit': 256, 'threads_per_job': max_threads})
    with open(os.path.join(config_folder, 'sparse_lifted_neighborhood.config'), 'w') as f:
        json.dump(conf, f)

    lifted_labels_path = input_path
    lifted_labels_key = 'volumes/predictions/classes/flagella_and_microvilli'

    n_scales = 1
    max_jobs_mc = 4
    task = LiftedMulticutSegmentationWorkflow(input_path=input_path, input_key=input_key,
                                              ws_path=input_path, ws_key=ws_key,
                                              problem_path=exp_path,
                                              node_labels_key=assignment_key,
                                              output_path=input_path,
                                              output_key=seg_key,
                                              n_scales=n_scales,
                                              lifted_labels_path=lifted_labels_path,
                                              lifted_labels_key=lifted_labels_key,
                                              lifted_prefix='flagella_microvilli',
                                              nh_graph_depth=4,
                                              config_dir=config_folder,
                                              tmp_folder=tmp_folder,
                                              target=target,
                                              max_jobs=max_jobs,
                                              max_jobs_multicut=max_jobs_mc,
                                              skip_ws=True,
                                              mode=mode)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Lifted Multicut failed"


def workflow(target, max_jobs, max_threads):
    # write the global config
    global_config = MulticutSegmentationWorkflow.get_config()['global']
    os.makedirs('configs', exist_ok=True)
    shebang = "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python"
    global_config.update({'block_shape': 3 * [256],
                          'shebang': shebang})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_config, f)

    run_mc(target, max_jobs, max_threads)
    # run_lmc(target, max_jobs, max_threads, mode='same')
    # run_lmc(target, max_jobs, max_threads, mode='all')


if __name__ == '__main__':
    workflow('slurm', 300, 8)
