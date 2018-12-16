#! /bin/python

import os
import sys
import json
import numpy as np

import luigi
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class CostsFromNodeLabelsBase(luigi.Task):
    """ CostsFromNodeLabels base class
    """

    task_name = 'costs_from_node_labels'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    nh_path = luigi.Parameter()
    nh_key = luigi.Parameter()
    node_label_path = luigi.Parameter()
    node_label_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    prefix = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        # intra_label_cost: cost of lifted edge connecting two nodes with the same node label
        #                   by default we set this to be very attractive (6 is close to the usual max val for costs)
        # inter_label_cost: cost of lifted edge connecting two nodes with different node labels
        #                   by default we set this to be very repulsive
        config.update({'intra_label_cost': 6.,
                       'inter_label_cost': -6.})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        #
        with vu.file_reader(self.nh_path, 'r') as f:
            n_lifted_edges = f[self.nh_key].shape

        # chunk size = 64**3
        chunk_size = min(262144, n_lifted_edges)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=(n_lifted_edges,),
                              chunks=(edge_chunk_size,), compression='gzip',
                              dtype='float32')

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'nh_path': self.nh_path,
                       'nh_key': self.nh_key,
                       'node_label_path': self.node_label_path,
                       'node_label_key': self.node_label_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'chunk_size': chunk_size})

        edge_block_list = vu.blocks_in_volume([n_lifted_edges], [chunk_size])
        n_jobs = min(self.max_jobs, len(edge_block_list))
        # prime and run the jobs
        self.prepare_jobs(n_jobs, edge_block_list, config, self.prefix)
        self.submit_jobs(n_jobs, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.prefix)
        self.check_jobs(n_jobs, self.prefix)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.prefix))


class CostsFromNodeLabelsLocal(CostsFromNodeLabelsBase, LocalTask):
    """ CostsFromNodeLabels on local machine
    """
    pass


class CostsFromNodeLabelsSlurm(CostsFromNodeLabelsBase, SlurmTask):
    """ CostsFromNodeLabels on slurm cluster
    """
    pass


class CostsFromNodeLabelsLSF(CostsFromNodeLabelsBase, LSFTask):
    """ CostsFromNodeLabels on lsf cluster
    """
    pass


#
# Implementation
#


def _costs_for_edge_block(block_id, blocking,
                          ds_in, ds_out, node_labels,
                          inter_label_cost, intra_label_cost):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    id_begin, id_end = block.begin, block.end
    uv_ids = ds_in[id_begin:id_end]

    labels_a = node_labels[uv_ids[:, 0]]
    labels_b = node_labels[uv_ids[:, 1]]

    # make sure we don't have nodes without labels
    assert np.sum(labels_a == 0) == np.sum(labels_b == 0) == 0

    edge_costs = intra_label_cost * np.ones(len(uv_ids), dtype='float32')
    edge_costs[labels_a != labels_b] = inter_label_cost

    ds_out[id_begin:id_end] = edge_costs
    # log block success
    fu.log_block_success(block_id)


def costs_from_node_labels(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    nh_path = config['nh_path']
    nh_key = config['nh_key']
    node_label_path = config['node_label_path']
    node_label_key = config['node_label_key']
    output_path = config['output_path']
    output_key = config['output_key']
    chunk_size = config['chunk_size']

    inter_label_cost = config['inter_label_cost']
    intra_label_cost = config['intra_label_cost']

    block_list = config['block_list']

    with vu.file_reader(node_label_path, 'r') as f:
        node_labels = f[node_label_key][:]
    with vu.file_reader(nh_path) as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[nh_key]
        ds_out = f_out[output_path]

        n_lifted_edges = ds_in.shape[0]
        blocking = nt.blocking([0], [n_lifted_edges], [chunk_size])

        for block_id in block_list:
            _costs_for_edge_block(block_id, blocking,
                                  ds_in, ds_out, node_labels,
                                  inter_label_cost, intra_label_cost)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    costs_from_node_labels(job_id, path)
