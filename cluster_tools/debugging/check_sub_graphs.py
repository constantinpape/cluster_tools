#! /usr/bin/python

import os
import sys
import json
from concurrent import futures
from collections import ChainMap

import luigi
import numpy as np
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class CheckSubGraphsBase(luigi.Task):
    """ CheckSubGraphs base class
    """

    allow_retry = False
    task_name = 'check_sub_graphs'
    src_file = os.path.abspath(__file__)

    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    graph_block_prefix = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # we don't need any additional config besides the paths
        config = self.get_task_config()
        config.update({"ws_path": self.ws_path, "ws_key": self.ws_key,
                       "graph_block_prefix": self.graph_block_prefix, "block_shape": block_shape,
                       "tmp_folder": self.tmp_folder})
        shape = vu.get_shape(self.ws_path, self.ws_key)
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class CheckSubGraphsLocal(CheckSubGraphsBase, LocalTask):
    """
    CheckSubGraphs on local machine
    """
    pass


class CheckSubGraphsSlurm(CheckSubGraphsBase, SlurmTask):
    """
    CheckSubGraphs on slurm cluster
    """
    pass


class CheckSubGraphsLSF(CheckSubGraphsBase, LSFTask):
    """
    CheckSubGraphs on lsf cluster
    """
    pass


#
# Implementation
#

def check_block(block_id, blocking, ds, graph_block_prefix):
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    seg = ds[bb]
    nodes_seg = np.unique(seg)

    graph_path = graph_block_prefix + str(block_id)
    nodes = ndist.loadNodes(graph_path)

    same_len = len(nodes_seg) == len(nodes)
    if not same_len:
        return block_id

    same_nodes = np.allclose(nodes, nodes_seg)
    if not same_nodes:
        return block_id

    return None


def check_sub_graphs(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)
    ws_path = config['ws_path']
    ws_key = config['ws_key']
    graph_block_prefix = config['graph_block_prefix']
    block_shape = config['block_shape']
    block_list = config['block_list']
    tmp_folder = config['tmp_folder']

    with vu.file_reader(ws_path, 'r') as f:
        ds = f[ws_key]
        shape = list(ds.shape)
        blocking = nt.blocking([0, 0, 0], shape, block_shape)
        violating_blocks = [check_block(block_id, blocking,
                                      ds, graph_block_prefix)
                            for block_id in block_list]
        violating_blocks = [vb for vb in violating_blocks if vb is not None]
    save_path = os.path.join(tmp_folder, 'failed_blocks_job_%i.json' % job_id)
    with open(save_path, 'w') as f:
        json.dump(violating_blocks, f)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    check_sub_graphs(job_id, path)
