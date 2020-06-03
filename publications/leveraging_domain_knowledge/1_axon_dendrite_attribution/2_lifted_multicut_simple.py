import argparse
import json
import multiprocessing
import os

import luigi
import numpy as np
import z5py
import nifty.distributed as ndist
from cluster_tools.features import RegionFeaturesWorkflow
from cluster_tools.lifted_multicut import LiftedMulticutWorkflow
from cluster_tools.write import WriteLocal, WriteSlurm
from elf.segmentation.multicut import transform_probabilities_to_costs

LIFTED_PREFIX = 'axondendrite'


def write_result(path, ws_key, node_labels_key, out_key, tmp_folder, target, n_jobs):
    task = WriteLocal if target == 'local' else WriteSlurm

    t = task(tmp_folder=tmp_folder, config_dir=os.path.join(tmp_folder, 'configs'),
             max_jobs=n_jobs,
             input_path=path, input_key=ws_key,
             output_path=path, output_key=out_key,
             assignment_path=path, assignment_key=node_labels_key,
             identifier=LIFTED_PREFIX)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def region_feats(path, ws_key, pmap_key, out_path,
                 tmp_folder, prefix, target, n_jobs):

    feat_key = f'region_features/{prefix}'
    task = RegionFeaturesWorkflow
    t = task(max_jobs=n_jobs, target=target,
             tmp_folder=tmp_folder, config_dir=os.path.join(tmp_folder, 'configs'),
             input_path=path, input_key=pmap_key,
             labels_path=path, labels_key=ws_key,
             output_path=out_path, output_key=feat_key,
             prefix=prefix)
    ret = luigi.build([t], local_scheduler=True)
    assert ret


def compute_lifted_nh(path, exp_path, semantic_label_key, lifted_nh_key, n_jobs,
                      ves_assignment_threshold, den_assignment_threshold):

    with z5py.File(exp_path, 'r') as f:
        if lifted_nh_key in f:
            return

    # map the region features to node labels
    with z5py.File(exp_path, 'r') as f:
        ves_feats = f['region_features/vesicles'][:, 1]
        den_feats = f['region_features/dendrites'][:, 1]
        print(ves_feats.min(), ves_feats.max())
        print(den_feats.min(), den_feats.max())

    node_labels = np.zeros(len(ves_feats), dtype='uint64')
    node_labels[ves_feats > ves_assignment_threshold] = 1
    assert np.sum(node_labels == 1) > 0
    node_labels[den_feats > den_assignment_threshold] = 2

    assert np.sum(node_labels == 2) > 0
    assert np.sum(node_labels == 1) > 0

    with z5py.File(path, 'a') as f:
        ds = f.require_dataset(semantic_label_key, shape=node_labels.shape,
                               compression='gzip', chunks=node_labels.shape, dtype=node_labels.dtype)
        ds[:] = node_labels

    # compute the lifted graph
    graph_key = 's0/graph'
    graph_depth = 10
    mode = 'different'
    node_ignore_label = 0
    ndist.computeLiftedNeighborhoodFromNodeLabels(exp_path, graph_key,
                                                  path, semantic_label_key,
                                                  exp_path, lifted_nh_key,
                                                  graph_depth, n_jobs,
                                                  mode, node_ignore_label)


def map_to_lifted_costs(path, exp_path, semantic_label_key, lifted_nh_key, lifted_cost_key, weight=1.5):
    with z5py.File(exp_path, 'r') as f:
        lifted_uvs = f[lifted_nh_key][:]
        ves_feats = f['region_features/vesicles'][:, 1]
        den_feats = f['region_features/dendrites'][:, 1]

    with z5py.File(path, 'r') as f:
        node_labels = f[semantic_label_key][:]

    node_feats = np.zeros(len(node_labels), dtype='float32')
    node_feats[node_labels == 1] = ves_feats[node_labels == 1]
    node_feats[node_labels == 2] = den_feats[node_labels == 2]

    lifted_costs = node_feats[lifted_uvs[:, 0]] * node_feats[lifted_uvs[:, 1]]
    lifted_costs = transform_probabilities_to_costs(lifted_costs)
    lifted_costs *= weight

    with z5py.File(exp_path, 'a') as f:
        ds = f.require_dataset(lifted_cost_key, shape=lifted_costs.shape, compression='gzip',
                               chunks=lifted_costs.shape, dtype=lifted_costs.dtype)
        ds[:] = lifted_costs


def make_lifted_problem(path, exp_path,
                        ws_key, ves_pmap, dendrite_pmap,
                        tmp_folder, target, n_jobs):
    region_feats(path, ws_key, ves_pmap, exp_path,
                 tmp_folder, 'vesicles', target, n_jobs)
    region_feats(path, ws_key, dendrite_pmap, exp_path,
                 tmp_folder, 'dendrites', target, n_jobs)

    ves_assignment_threshold = .5
    den_assignment_threshold = .5

    semantic_label_key = 'node_labels/semantic'
    lifted_nh_key = f's0/lifted_nh_{LIFTED_PREFIX}'
    compute_lifted_nh(path, exp_path,
                      semantic_label_key, lifted_nh_key,
                      n_jobs, ves_assignment_threshold, den_assignment_threshold)

    lifted_cost_key = f's0/lifted_costs_{LIFTED_PREFIX}'
    map_to_lifted_costs(path, exp_path, semantic_label_key, lifted_nh_key, lifted_cost_key)


def solve_lifted_problem(path, exp_path,
                         ws_key, node_labels_key, out_key,
                         tmp_folder, target, n_jobs):
    task = LiftedMulticutWorkflow

    t = task(tmp_folder=tmp_folder, config_dir=os.path.join(tmp_folder, 'configs'),
             target=target, max_jobs=n_jobs,
             problem_path=exp_path,
             assignment_path=path, assignment_key=node_labels_key,
             n_scales=1, lifted_prefix=LIFTED_PREFIX)
    ret = luigi.build([t], local_scheduler=True)
    assert ret

    # write the result
    write_result(path, ws_key, node_labels_key, out_key,
                 tmp_folder, target, n_jobs)


def run_lifted_multicut(target, n_jobs):
    task = LiftedMulticutWorkflow

    input_path = './data.n5'
    tmp_folder = './tmp_mc'
    exp_path = os.path.join(tmp_folder, 'exp_data.n5')
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    ws_key = 'volumes/segmentation/watershed'
    out_key = 'volumes/segmentation/lifted_multicut'
    node_labels_key = 'node_labels/lifted_multicut'

    ves_pmap = 'probs/vesicles'
    dendrite_pmap = 'probs/dendrites'

    configs = task.get_config()

    global_config = configs['global']
    global_config.update({'block_shape': [140, 140, 140]})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    make_lifted_problem(input_path, exp_path,
                        ws_key, ves_pmap, dendrite_pmap,
                        tmp_folder, target, n_jobs)

    solve_lifted_problem(input_path, exp_path,
                         ws_key, node_labels_key, out_key,
                         tmp_folder, target, n_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='local', type=str)
    parser.add_argument('--n_jobs', default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()
    run_lifted_multicut(args.target, args.n_jobs)
