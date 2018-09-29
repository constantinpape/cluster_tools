#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Graph Tasks
#


class InitialSubGraphsBase(luigi.Task):
    """ InitialSubGraph base class
    """

    task_name = 'initial_sub_graphs'
    src_file = os.path.abspath(__file__)

    # input volumes and graph
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'ignore_label': True})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'graph_path': self.graph_path, 'block_shape': block_shape})

        # make graph file and write shape as attribute
        shape = vu.get_shape(self.input_path, self.input_key)
        with vu.file_reader(self.graph_path) as f:
            f.attrs['shape'] = shape

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class InitialSubGraphsLocal(InitialSubGraphsBase, LocalTask):
    """ InitialSubGraphs on local machine
    """
    pass


class InitialSubGraphsSlurm(InitialSubGraphsBase, SlurmTask):
    """ InitialSubGraphs on slurm cluster
    """
    pass


class InitialSubGraphsLSF(InitialSubGraphsBase, LSFTask):
    """ InitialSubGraphs on lsf cluster
    """
    pass


#
# Implementation
#


def _graph_block(block_id, blocking, input_path, input_key, graph_path,
                 ignore_label):
    fu.log("start processing block %i" % block_id)
    halo = [1, 1, 1]
    block = blocking.getBlockWithHalo(block_id, halo)
    outer_block, inner_block = block.outerBlock, block.innerBlock
    # we only need the halo into one direction,
    # hence we use the outer-block only for the end coordinate
    begin = inner_block.begin
    end = outer_block.end

    block_key = 's0/sub_graphs/block_%i' % block_id
    ndist.computeMergeableRegionGraph(input_path, input_key,
                                      begin, end,
                                      graph_path, block_key,
                                      ignore_label)
    # log block success
    fu.log_block_success(block_id)


# TODO make work with hdf5 as well
def initial_sub_graphs(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    block_shape = config['block_shape']
    block_list = config['block_list']
    graph_path = config['graph_path']
    ignore_label = config.get('ignore_label', True)

    shape = vu.get_shape(input_path, input_key)
    blocking = nt.blocking(roiBegin=[0, 0, 0],
                           roiEnd=list(shape),
                           blockShape=list(block_shape))

    for block_id in block_list:
        _graph_block(block_id, blocking, input_path, input_key, graph_path,
                     ignore_label)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    initial_sub_graphs(job_id, path)
