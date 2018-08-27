#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env/bin/python

import os
import sys
import json

import luigi

from cluster_tools.multicut import MulticutWorkflow
from cluster_tools.multicut.solve_subproblems import SolveSubproblemsLocal


def multicut_example(shebang):
    graph_path = '/home/cpape/Work/data/isbi2012/cluster_example/graph.n5'
    costs_path = '/home/cpape/Work/data/isbi2012/cluster_example/costs.n5'
    output_path = '/home/cpape/Work/data/isbi2012/cluster_example/node_labels.n5'
    merged_out = '/home/cpape/Work/data/isbi2012/cluster_example/problems.n5'

    tmp_folder = './tmp'
    config_folder = './configs'

    max_jobs = 8
    global_conf = SolveSubproblemsLocal.default_global_config()
    global_conf.update({'shebang': shebang,
                        'block_shape': (10, 256, 256)})
    with open('./configs/global.config', 'w') as f:
        json.dump(global_conf, f)

    ret = luigi.build([MulticutWorkflow(graph_path=graph_path,
                                        graph_key='graph',
                                        costs_path=costs_path,
                                        costs_key='costs',
                                        n_scales=1,
                                        merged_problem_path=merged_out,
                                        output_path=output_path,
                                        output_key='node_labels',
                                        config_dir=config_folder,
                                        tmp_folder=tmp_folder,
                                        target='local',
                                        max_jobs=max_jobs)], local_scheduler=True)


if __name__ == '__main__':
    shebang = '#! /home/cpape/Work/software/conda/miniconda3/envs/affogato/bin/python'
    multicut_example(shebang)
