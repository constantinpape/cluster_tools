#! /usr/bin/python

import os
import sys
import argparse
import json

import numpy as np
import luigi
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class EdgeLabelsBase(luigi.Task):
    """ Edge labels base class
    """

    task_name = 'edge_labels'
    src_file = os.path.abspath(__file__)
    # retry is too complecated for now ...
    allow_retry = False

    # input and output volumes
    node_labels_path = luigi.Parameter()
    node_labels_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'ignore_label_gt': False})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, _, _, _ = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the task config
        config.update({'node_labels_path': self.node_labels_path, 'node_labels_key': self.node_labels_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'graph_path': self.graph_path, 'graph_key': self.graph_key})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class EdgeLabelsLocal(EdgeLabelsBase, LocalTask):
    """ EdgeLabels on local machine
    """
    pass


class EdgeLabelsSlurm(EdgeLabelsBase, SlurmTask):
    """ EdgeLabels on slurm cluster
    """
    pass


class EdgeLabelsLSF(EdgeLabelsBase, LSFTask):
    """ EdgeLabels on lsf cluster
    """
    pass


#
# Implementation
#


# TODO parallelize
def edge_labels(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    output_path = config['output_path']
    output_key = config['output_key']
    graph_path = config['graph_path']
    graph_key = config['graph_key']
    node_labels_path = config['node_labels_path']
    node_labels_key = config['node_labels_key']
    ignore_label_gt = config.get('ignore_label_gt', False)

    # load the node labels
    with vu.file_reader(node_labels_path, 'r') as f:
        node_labels = f[node_labels_key][:]

    # load the uv ids and check
    with vu.file_reader(graph_path, 'r') as f:
        uv_ids = f[graph_key]['edges'][:]

    lu = node_labels[uv_ids[:, 0]]
    lv = node_labels[uv_ids[:, 1]]
    edge_labels = (lu != lv).astype('int8')
    if ignore_label_gt:
        ignore_mask = np.logical_or(lu == 0, lv == 0)
        edge_labels[ignore_mask] = -1

    n_edges = len(edge_labels)
    chunks = (min(262144, n_edges),)
    with vu.file_reader(output_path) as f:
        f.create_dataset(output_key, data=edge_labels,
                         chunks=chunks, compression='gzip')

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    edge_labels(job_id, path)
