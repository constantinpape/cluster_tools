#! /usr/bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class MergeRegionFeaturesBase(luigi.Task):
    """ Merge edge feature base class
    """

    task_name = 'merge_region_features'
    src_file = os.path.abspath(__file__)
    # retry is too complecated for now ...
    allow_retry = False

    # input and output volumes
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    number_of_labels = luigi.IntParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        chunk_size = min(10000, self.number_of_labels)

        # require the output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, dtype='float32', shape=(self.number_of_labels,),
                              chunks=(chunk_size,), compression='gzip')

        # temporary output dataset
        tmp_path = os.path.join(self.tmp_folder, 'region_features_tmp.n5')
        tmp_key = 'block_feats'
        # update the task config
        config.update({'output_path': self.output_path, 'output_key': self.output_key,
                       'tmp_path': tmp_path, 'tmp_key': tmp_key,
                       'node_chunk_size': chunk_size})

        node_block_list = vu.blocks_in_volume([self.number_of_labels], [chunk_size])

        n_jobs = min(len(node_block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, node_block_list, config, consecutive_blocks=True)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class MergeRegionFeaturesLocal(MergeRegionFeaturesBase, LocalTask):
    """ MergeRegionFeatures on local machine
    """
    pass


class MergeRegionFeaturesSlurm(MergeRegionFeaturesBase, SlurmTask):
    """ MergeRegionFeatures on slurm cluster
    """
    pass


class MergeRegionFeaturesLSF(MergeRegionFeaturesBase, LSFTask):
    """ MergeRegionFeatures on lsf cluster
    """
    pass


#
# Implementation
#

def _extract_and_merge_region_features(blocking, ds_in, ds, node_begin, node_end):
    fu.log("processing node range %i to %i" % (node_begin, node_end))
    out_features = np.zeros(node_end - node_begin, dtype='float32')
    out_counts = np.zeros(node_end - node_begin, dtype='float32')

    chunks = ds_in.chunks
    for block_id in range(blocking.numberOfBlocks):
        block = blocking.getBlock(block_id)
        chunk_id = tuple(beg // ch for beg, ch in zip(block.begin, chunks))

        # load the data
        data = ds_in.read_chunk(chunk_id)
        if data is None:
            continue

        # TODO support more features
        # extract ids and features
        ids = data[::3].astype('uint64')
        counts = data[1::3]
        mean = data[2::3]

        # check if any ids overlap with our id range
        overlap_mask = np.logical_and(ids >= node_begin,
                                      ids < node_end)
        if np.sum(overlap_mask) == 0:
            continue

        overlapping_ids = ids[overlap_mask]
        overlapping_ids -= node_begin
        overlapping_counts = counts[overlap_mask]
        overlapping_mean = mean[overlap_mask]

        # calculate cumulative moving average
        prev_counts = out_counts[overlapping_ids]
        tot_counts = (prev_counts + overlapping_counts)
        out_feats = (overlapping_counts * overlapping_mean + prev_counts * out_features[overlapping_ids]) / tot_counts
        out_features[overlapping_ids] = out_feats
        out_counts[overlapping_ids] += overlapping_counts

    out_features[np.isnan(out_features)] = 0.
    ds[node_begin:node_end] = out_features


def merge_region_features(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)
    output_path = config['output_path']
    output_key = config['output_key']
    tmp_path = config['tmp_path']
    tmp_key = config['tmp_key']
    node_block_list = config['block_list']
    node_chunk_size = config['node_chunk_size']

    with vu.file_reader(output_path) as f,\
            vu.file_reader(tmp_path) as f_in:

        ds_in = f_in[tmp_key]
        ds = f[output_key]
        n_nodes = ds.shape[0]

        node_blocking = nt.blocking([0], [n_nodes], [node_chunk_size])
        node_begin = node_blocking.getBlock(node_block_list[0]).begin[0]
        node_end = node_blocking.getBlock(node_block_list[-1]).end[0]

        shape = list(ds_in.shape)
        chunks = list(ds_in.chunks)
        blocking = nt.blocking([0, 0, 0], shape, chunks)

        _extract_and_merge_region_features(blocking, ds_in, ds, node_begin, node_end)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_region_features(job_id, path)
