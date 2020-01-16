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

class MergeSubGraphsBase(luigi.Task):
    """ MergeSubGraph base class
    """

    task_name = 'merge_sub_graphs'
    src_file = os.path.abspath(__file__)

    # input volumes and graph
    graph_path = luigi.Parameter()
    scale = luigi.IntParameter()
    output_key = luigi.Parameter(default='')
    merge_complete_graph = luigi.BoolParameter(default=False)
    # dependency
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def _initialize_datasets(self, shape, block_shape):
        f = vu.file_reader(self.graph_path)
        output_key = 's%i/sub_graphs' % self.scale
        node_key = os.path.join(output_key, 'nodes')
        f.require_dataset(node_key, shape=shape, chunks=block_shape,
                          compression='gzip', dtype='uint64')
        edge_key = os.path.join(output_key, 'edges')
        f.require_dataset(edge_key, shape=shape, chunks=block_shape,
                          compression='gzip', dtype='uint64')

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the watershed config
        config = self.get_task_config()

        # get the shape
        with vu.file_reader(self.graph_path, 'r') as f:
            shape = f.attrs['shape']

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'graph_path': self.graph_path, 'block_shape': block_shape,
                       'scale': self.scale, 'merge_complete_graph': self.merge_complete_graph,
                       'output_key': self.output_key})

        factor = 2**self.scale
        block_shape = tuple(sh * factor for sh in block_shape)
        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        if self.merge_complete_graph:
            n_jobs = 1
        else:
            n_jobs = min(len(block_list), self.max_jobs)
            self._initialize_datasets(shape, block_shape)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_s%i.log' % self.scale))



class MergeSubGraphsLocal(MergeSubGraphsBase, LocalTask):
    """ MergeSubGraphs on local machine
    """
    pass


class MergeSubGraphsSlurm(MergeSubGraphsBase, SlurmTask):
    """ MergeSubGraphs on slurm cluster
    """
    pass


class MergeSubGraphsLSF(MergeSubGraphsBase, LSFTask):
    """ MergeSubGraphs on lsf cluster
    """
    pass


#
# Implementation
#


def _merge_graph(graph_path, output_key, scale,
                 block_list, blocking, shape, n_threads):
    subgraph_key = 's%i/sub_graphs' % scale
    ndist.mergeSubgraphs(graph_path,
                         subgraphKey=subgraph_key,
                         blockIds=block_list,
                         outKey=output_key,
                         numberOfThreads=n_threads,
                         serializeToVarlen=False)
    with vu.file_reader(graph_path) as f:
        f[output_key].attrs['shape'] = shape


def _merge_subblocks(block_id, blocking, previous_blocking, graph_path, scale):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    input_key = 's%i/sub_graphs' % (scale - 1,)
    output_key = 's%i/sub_graphs' % scale
    block_list = previous_blocking.getBlockIdsInBoundingBox(roiBegin=block.begin,
                                                            roiEnd=block.end,
                                                            blockHalo=[0, 0, 0])
    ndist.mergeSubgraphs(graph_path,
                         subgraphKey=input_key,
                         blockIds=block_list.tolist(),
                         outKey=output_key, serializeToVarlen=True)
    # log block success
    fu.log_block_success(block_id)


def merge_sub_graphs(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    scale = config['scale']
    initial_block_shape = config['block_shape']
    graph_path = config['graph_path']
    merge_complete_graph = config['merge_complete_graph']
    block_list = config['block_list']

    with vu.file_reader(graph_path, 'r') as f:
        shape = f.attrs['shape']
    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]
    blocking = nt.blocking(roiBegin=[0, 0, 0],
                           roiEnd=list(shape),
                           blockShape=block_shape)

    if merge_complete_graph:
        fu.log("merge complete graph at scale %i" % scale)
        output_key = config['output_key']
        assert output_key != ''
        n_threads = config['threads_per_job']
        _merge_graph(graph_path, output_key, scale,
                     block_list, blocking, shape, n_threads)

    else:
        fu.log("merging subgraphs at scale %i" % scale)
        previous_factor = 2**(scale - 1)
        previous_block_shape = [previous_factor * bs for bs in initial_block_shape]
        previous_blocking = nt.blocking(roiBegin=[0, 0, 0],
                                        roiEnd=list(shape),
                                        blockShape=previous_block_shape)
        for block_id in block_list:
            _merge_subblocks(block_id, blocking, previous_blocking,
                             graph_path, output_key, scale)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_sub_graphs(job_id, path)
