#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi

from cluster_tools.costs import EdgeCostsWorkflow
from cluster_tools.costs.probs_to_costs import ProbsToCostsLocal


def features_example(shebang):
    features_path = '/home/cpape/Work/data/isbi2012/cluster_example/features.n5'
    features_key = 'features'
    output_path = '/home/cpape/Work/data/isbi2012/cluster_example/costs.n5'

    input_key = 'volumes/affinities'
    labels_key = 'volumes/watershed'

    tmp_folder = './tmp'
    config_folder = './configs'

    max_jobs = 8
    global_conf = ProbsToCostsLocal.default_global_config()
    global_conf.update({'shebang': shebang})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_conf, f)

    ret = luigi.build([EdgeCostsWorkflow(features_path=features_path,
                                         features_key=features_key,
                                         output_path=output_path,
                                         output_key='costs',
                                         config_dir=config_folder,
                                         tmp_folder=tmp_folder,
                                         target='local',
                                         max_jobs=max_jobs)], local_scheduler=True)


if __name__ == '__main__':
    shebang = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
    features_example(shebang)
