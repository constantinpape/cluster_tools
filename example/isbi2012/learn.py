#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi
import z5py

from cluster_tools import LearningWorkflow


def run(shebang):
    example_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'

    input_dict = {'train': (example_path, 'volumes/affinities')}
    labels_dict = {'train': (example_path, 'volumes/watersheds')}
    groundtruth_dict = {'train': (example_path, 'volumes/groundtruth')}

    max_jobs = 8

    configs = LearningWorkflow.get_config()

    global_conf = configs['global']
    global_conf.update({'shebang': shebang,
                        'block_shape': (25, 256, 256)})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_conf, f)

    feat_config = configs['block_edge_features']
    feat_config.update({'filters': ['gaussianSmoothing', 'laplacianOfGaussian'],
                        'sigmas': [1., 2., 4.], 'apply_in_2d': True})
    with open('./configs/block_edge_features.config', 'w') as f:
        json.dump(feat_config, f)

    rf_path = './rf.pkl'
    ret = luigi.build([LearningWorkflow(input_dict=input_dict,
                                        labels_dict=labels_dict,
                                        groundtruth_dict=groundtruth_dict,
                                        output_path=rf_path,
                                        config_dir='./configs',
                                        tmp_folder='./tmp',
                                        target='local',
                                        max_jobs=max_jobs)], local_scheduler=True)
    if ret:
        assert os.path.exists(rf_path)
        print("Have rf")


if __name__ == '__main__':
    shebang = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
    run(shebang)
