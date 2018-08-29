#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi
import z5py

from cluster_tools import LearningWorkflow


def learn_rf():

    def get_path(block_id):
        return '/g/kreshuk/data/arendt/platyneris_v1/membrane_training_data/predictions/train_block_0%i_unet_lr_v3.n5' % block_id

    blocks = (7, 8)
    input_dict = {'train_%i' % block_id: (get_path(block_id), 'volumes/affinities') for block_id in blocks}
    labels_dict = {'train_%i' % block_id: (get_path(block_id), 'volumes/watershed') for block_id in blocks}
    groundtruth_dict = {'train_%i' % block_id: (get_path(block_id), 'volumes/groundtruth') for block_id in blocks}

    max_jobs = 8

    configs = LearningWorkflow.get_config()

    rf_conf = configs['learn_rf']
    rf_conf.update({'n_trees': 200})
    with open('./config/learn_rf.config', 'w') as f:
        json.dump(rf_conf, f)

    edge_labels_conf = configs['edge_labels']
    edge_labels_conf.update({'ignore_label_gt': True})
    with open('./config/edge_labels.config', 'w') as f:
        json.dump(edge_labels_conf, f)

    feats_conf = configs['block_edge_features']
    feats_conf.update({'filters': ['gaussianSmoothing'],
                       'sigmas': [(0.5, 1., 1.), (1., 2., 2.), (2., 4., 4.), (4., 8., 8.)],
                       'halo': (8, 16, 16)})
    with open('./config/block_edge_features.config', 'w') as f:
        json.dump(feats_conf, f)

    root = os.path.split(get_path(0))[0]
    rf_path = os.path.join(root, 'rf_v1.pkl')
    ret = luigi.build([LearningWorkflow(input_dict=input_dict,
                                        labels_dict=labels_dict,
                                        groundtruth_dict=groundtruth_dict,
                                        output_path=rf_path,
                                        config_dir='./config',
                                        tmp_folder='./tmp_learn',
                                        target='local',
                                        max_jobs=max_jobs)], local_scheduler=True)
    if ret:
        assert os.path.exists(rf_path)
        print("Have rf")


if __name__ == '__main__':
    learn_rf()
