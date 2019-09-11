#! /usr/bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty.distributed as ndist
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class MergeEdgeFeaturesBase(luigi.Task):
    """ Merge edge feature base class
    """

    task_name = 'merge_edge_features'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def _read_num_features(self, block_ids):
        n_feats = None
        with vu.file_reader(self.output_path) as f:
            for block_id in block_ids:
                block_key = os.path.join('blocks', 'block_%i' % block_id)
                block_path = os.path.join(self.output_path, block_key)
                if not os.path.exists(block_path):
                    continue
                n_feats = f[block_key].shape[1]
                break
        assert n_feats is not None, "No valid feature block found"
        return n_feats

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # get the number of graph edges and the volume shape
        with vu.file_reader(self.graph_path) as f:
            shape = f.attrs['shape']
            n_edges = f[self.graph_key].attrs['numberOfEdges']

        # if we don't have a roi, we only serialize the number of blocks
        # otherwise we serialize the blocks in roi
        if roi_begin is None:
            block_ids = nt.blocking([0, 0, 0], shape, block_shape).numberOfBlocks
        else:
            block_ids = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)

        # chunk size = 64**3
        chunk_size = min(262144, n_edges)

        # get the number of features from sub-feature block
        n_features = self._read_num_features(range(block_ids) if isinstance(block_ids, int)
                                             else block_ids)

        # require the output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, dtype='float64', shape=(n_edges, n_features),
                              chunks=(chunk_size, 1), compression='gzip')

        # update the task config
        config.update({'graph_path': self.graph_path,
                       'block_prefix': os.path.join('s0', 'sub_graphs', 'block_'),
                       'in_path': self.output_path, 'in_prefix': 'blocks/block_',
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'edge_chunk_size': chunk_size, 'block_ids': block_ids,
                       'n_edges': n_edges})

        edge_block_list = vu.blocks_in_volume([n_edges], [chunk_size])

        n_jobs = min(len(edge_block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, edge_block_list, config,
                          consecutive_blocks=True)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class MergeEdgeFeaturesLocal(MergeEdgeFeaturesBase, LocalTask):
    """ MergeEdgeFeatures on local machine
    """
    pass


class MergeEdgeFeaturesSlurm(MergeEdgeFeaturesBase, SlurmTask):
    """ MergeEdgeFeatures on slurm cluster
    """
    pass


class MergeEdgeFeaturesLSF(MergeEdgeFeaturesBase, LSFTask):
    """ MergeEdgeFeatures on lsf cluster
    """
    pass


#
# Implementation
#


def merge_edge_features(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    graph_path = config['graph_path']
    block_prefix = config['block_prefix']
    in_path = config['in_path']
    in_prefix = config['in_prefix']
    output_path = config['output_path']
    output_key = config['output_key']
    n_threads = config['threads_per_job']
    edge_block_list = config['block_list']
    edge_chunk_size = config['edge_chunk_size']
    block_ids = config['block_ids']

    # assert that the edge block list is consecutive
    diff_list = np.diff(edge_block_list)
    assert (diff_list == 1).all()

    n_edges = config['n_edges']
    edge_blocking = nt.blocking([0], [n_edges], [edge_chunk_size])
    edge_begin = edge_blocking.getBlock(edge_block_list[0]).begin[0]
    edge_end = edge_blocking.getBlock(edge_block_list[-1]).end[0]

    # the block list might either be the number of blocks or a list of blocks
    block_ids = list(range(block_ids)) if isinstance(block_ids, int) else block_ids

    ndist.mergeFeatureBlocks(graph_path, block_prefix,
                             in_path, in_prefix,
                             output_path, output_key,
                             blockIds=block_ids,
                             edgeIdBegin=edge_begin,
                             edgeIdEnd=edge_end,
                             numberOfThreads=n_threads)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_edge_features(job_id, path)
