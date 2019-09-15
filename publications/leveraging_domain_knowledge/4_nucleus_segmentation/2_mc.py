#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
import json
import luigi

from cluster_tools import MulticutSegmentationWorkflow


def run_mc(timepoint, skip_ws):

    target = 'local'
    max_jobs = 48
    max_threads = 16

    config_folder = './configs'
    os.makedirs(config_folder, exist_ok=True)
    tmp_folder = './tmp_mc_t%03i' % timepoint

    path = '/g/kreshuk/pape/Work/data/lifted_priors/plant_data/t%03i.n5' % timepoint
    exp_path = os.path.join('/g/kreshuk/pape/Work/data/lifted_priors/plant_data/exp_data',
                            'mc_t%03i.n5' % timepoint)

    input_key = 'volumes/predictions/boundaries'
    ws_key = 'volumes/segmentation/watershed'
    assignment_key = 'node_labels/multicut'
    out_key = 'volumes/segmentation/multicut'

    configs = MulticutSegmentationWorkflow.get_config()

    global_conf = configs['global']
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    block_shape = [50, 512, 512]
    global_conf.update({'shebang': shebang, 'block_shape': block_shape})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    conf = configs['watershed']
    conf.update({'threshold': .25, 'apply_dt_2d': False, 'apply_ws_2d': False,
                 'sigma_seeds': 2., 'sigma_weights': 2., 'alpha': .9,
                 'min_seg_size': 100, 'non_maximum_suppression': True})
    with open(os.path.join(config_folder, 'watershed.config'), 'w') as f:
        json.dump(conf, f)

    exponent = 1.
    weight_edges = True
    costs_config = configs['probs_to_costs']
    costs_config.update({'weight_edges': weight_edges,
                         'weighting_exponent': exponent})
    with open(os.path.join(config_folder, 'probs_to_costs.config'), 'w') as f:
        json.dump(costs_config, f)

    # set number of threads for sum jobs
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'map_edge_ids',
             'reduce_problem', 'solve_global']
    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_threads,
                       'agglomerator': 'kernighan-lin'})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    n_scales = 0
    task = MulticutSegmentationWorkflow(tmp_folder=tmp_folder, config_dir=config_folder,
                                        target=target,
                                        max_jobs=max_jobs, max_jobs_multicut=1,
                                        input_path=path, input_key=input_key,
                                        ws_path=path, ws_key=ws_key,
                                        problem_path=exp_path,
                                        node_labels_key=assignment_key,
                                        output_path=path, output_key=out_key,
                                        n_scales=n_scales,
                                        sanity_checks=False,
                                        skip_ws=skip_ws)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Multicut failed"


if __name__ == '__main__':
    skip_ws = True
    for timepoint in (45, 49):
        run_mc(timepoint=timepoint, skip_ws=skip_ws)
