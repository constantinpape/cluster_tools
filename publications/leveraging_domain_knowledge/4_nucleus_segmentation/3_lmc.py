#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import os
import json
import luigi

from cluster_tools import LiftedMulticutSegmentationWorkflow


def run_lmc(timepoint, only_repulsions):

    target = 'local'
    max_jobs = 48
    max_threads = 16

    name = 't%03i_%s' % (timepoint, 'repulsive' if only_repulsions else 'all')

    config_folder = './configs'
    os.makedirs(config_folder, exist_ok=True)
    tmp_folder = './tmp_lmc_%s' % name

    path = '/g/kreshuk/pape/Work/data/lifted_priors/plant_data/t%03i.n5' % timepoint
    exp_path = os.path.join('/g/kreshuk/pape/Work/data/lifted_priors/plant_data/exp_data',
                            'lmc_%s.n5' % name)

    input_key = 'volumes/predictions/boundaries'
    ws_key = 'volumes/segmentation/watershed'
    nucleus_seg_key = 'volumes/segmentation/nuclei'
    if only_repulsions:
        assignment_key = 'node_labels/lifted_multicut_repulsive'
        out_key = 'volumes/segmentation/lifted_multicut_repulsive'
    else:
        assignment_key = 'node_labels/lifted_multicut_all'
        out_key = 'volumes/segmentation/lifted_multicut_all'

    configs = LiftedMulticutSegmentationWorkflow.get_config()

    global_conf = configs['global']
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    block_shape = [50, 512, 512]
    global_conf.update({'shebang': shebang, 'block_shape': block_shape})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_conf, f)

    exponent = 1.
    weight_edges = True
    costs_config = configs['probs_to_costs']
    costs_config.update({'weight_edges': weight_edges,
                         'weighting_exponent': exponent})
    with open(os.path.join(config_folder, 'probs_to_costs.config'), 'w') as f:
        json.dump(costs_config, f)

    conf = configs['sparse_lifted_neighborhood']
    conf.update({'threads_per_job': max_threads})
    with open(os.path.join(config_folder, 'sparse_lifted_neighborhood.config'), 'w') as f:
        json.dump(conf, f)

    # set number of threads for sum jobs
    tasks = ['merge_sub_graphs', 'merge_edge_features', 'map_edge_ids',
             'reduce_lifted_problem', 'solve_lifted_global']
    for tt in tasks:
        config = configs[tt]
        config.update({'threads_per_job': max_threads,
                       'agglomerator': 'kernighan-lin'})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    tasks = ['block_node_labels', 'merge_node_labels']
    for tt in tasks:
        config = configs[tt]
        config.update({"time_limit": 60, "mem_limit": 16})
        with open(os.path.join(config_folder, '%s.config' % tt), 'w') as f:
            json.dump(config, f)

    # set mode to 'different' to get only repulsive edges
    n_scales = 0
    # thresh = .15
    thresh = None
    mode = 'different' if only_repulsions else 'all'
    task = LiftedMulticutSegmentationWorkflow(tmp_folder=tmp_folder, config_dir=config_folder, target=target,
                                              max_jobs=max_jobs, max_jobs_multicut=1,
                                              input_path=path, input_key=input_key,
                                              ws_path=path, ws_key=ws_key,
                                              problem_path=exp_path,
                                              node_labels_key=assignment_key,
                                              output_path=path, output_key=out_key,
                                              n_scales=n_scales, sanity_checks=False, mode=mode,
                                              lifted_labels_path=path, lifted_labels_key=nucleus_seg_key,
                                              node_ignore_label=0, label_ignore_label=0,
                                              label_overlap_threshold=thresh,
                                              skip_ws=True, nh_graph_depth=32, lifted_prefix='nucleus_segmentation')
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Lifted multicut failed"


if __name__ == '__main__':
    for timepoint in (45, 49):
        run_lmc(timepoint, True)
        run_lmc(timepoint, False)

    # don't run with all edges for now
    # run_lmc(timepoint, False)
