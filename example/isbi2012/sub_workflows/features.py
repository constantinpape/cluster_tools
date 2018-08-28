#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi

from cluster_tools.features import EdgeFeaturesWorkflow
from cluster_tools.features.block_edge_features import BlockEdgeFeaturesLocal

NEAREST_OFFSETS = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]


def features_example(shebang):
    input_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'
    labels_path = '/home/cpape/Work/data/isbi2012/cluster_example/isbi_train.n5'
    graph_path = '/home/cpape/Work/data/isbi2012/cluster_example/graph.n5'
    output_path = '/home/cpape/Work/data/isbi2012/cluster_example/features.n5'

    input_key = 'volumes/affinities'
    labels_key = 'volumes/watersheds'

    tmp_folder = './tmp'
    config_folder = './configs'

    max_jobs = 8
    global_conf = BlockEdgeFeaturesLocal.default_global_config()
    global_conf.update({'shebang': shebang, 'block_shape': [10, 256, 256]})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_conf, f)

    task_config = BlockEdgeFeaturesLocal.default_task_config()
    task_config.update({'offsets': NEAREST_OFFSETS})
    with open('./configs/block_edge_features.config', 'w') as f:
        json.dump(task_config, f)

    ret = luigi.build([EdgeFeaturesWorkflow(input_path=input_path,
                                            input_key=input_key,
                                            labels_path=labels_path,
                                            labels_key=labels_key,
                                            graph_path=graph_path,
                                            graph_key='graph',
                                            output_path=output_path,
                                            output_key='features',
                                            config_dir=config_folder,
                                            tmp_folder=tmp_folder,
                                            target='local',
                                            max_jobs=max_jobs,
                                            max_jobs_merge=1)], local_scheduler=True)


if __name__ == '__main__':
    shebang = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
    features_example(shebang)
