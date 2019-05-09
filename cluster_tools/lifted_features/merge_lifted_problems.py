#! /bin/python

import os
import sys
import json
import numpy as np
import z5py

import luigi

import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Merge Lifted Problem Tasks
#

class MergeLiftedProblemsBase(luigi.Task):
    """ MergeLiftedProblems base class
    """

    task_name = 'merge_lifted_problems'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    path = luigi.Parameter()
    prefixs = luigi.IntParameter()
    out_prefix = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'path': self.path,
                       'prefixs': self.prefixs,
                       'out_prefix': self.out_prefix})

        # prime and run the jobs
        self.prepare_jobs(1, None, config, self.out_prefix)
        self.submit_jobs(1, self.out_prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.out_prefix)
        self.check_jobs(1, self.out_prefix)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.out_prefix))


class MergeLiftedProblemsLocal(MergeLiftedProblemsBase, LocalTask):
    """ MergeLiftedProblems on local machine
    """
    pass


class MergeLiftedProblemsSlurm(MergeLiftedProblemsBase, SlurmTask):
    """ MergeLiftedProblems on slurm cluster
    """
    pass


class MergeLiftedProblemsLSF(MergeLiftedProblemsBase, LSFTask):
    """ MergeLiftedProblems on lsf cluster
    """
    pass


#
# Implementation
#

def merge_lifted_problems(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    path = config['path']
    prefixs = config['prefixs']
    out_prefix = config['out_prefix']
    n_threads = config.get('threads_per_job', 1)

    f = z5py.File(path)
    edge_root = 's0/lifted_nh_%s'
    cost_root = 's0/lifted_costs_%s'

    edges = []
    costs = []
    for prefix in prefixs:
        edge_key = edge_root % prefix
        cost_key = cost_root % prefix

        ds_edges = f[edge_key]
        ds_edges.n_threads = n_threads
        this_edges = ds_edges[:]

        ds_costs = f[cost_key]
        ds_costs.n_threads = n_threads
        this_costs = ds_costs[:]

        assert len(this_costs) == len(this_edges)
        edges.append(this_edges)
        costs.append(this_costs)

    # TODO would be cleaner to
    # - sort the edges again
    # - see if any of the edges are duplicate and add up costs if they are
    edges = np.concatenate(edges, axis=0)
    costs = np.concatenate(costs, axis=0)

    edge_out_key = edge_root % out_prefix
    edge_chunks = (min(len(edges), 100000), 2)
    ds_edges_out = f.require_dataset(edge_out_key, shape=edges.shape, compression='gzip',
                                     dtype=edges.dtype, chunks=edge_chunks)
    ds_edges_out.n_threads = n_threads
    ds_edges_out[:] = edges

    cost_out_key = cost_root % out_prefix
    cost_chunks = (min(len(costs), 100000),)
    ds_costs_out = f.require_dataset(cost_out_key, shape=costs.shape, compression='gzip',
                                     dtype=costs.dtype, chunks=cost_chunks)
    ds_costs_out.n_threads = n_threads
    ds_costs_out[:] = costs

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_lifted_problems(job_id, path)
