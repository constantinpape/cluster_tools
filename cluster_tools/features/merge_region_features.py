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
    prefix = luigi.Parameter(default='')

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        chunk_size = min(10000, self.number_of_labels)

        # temporary output dataset
        tmp_path = os.path.join(self.tmp_folder, 'region_features_tmp.n5')
        tmp_key = 'block_feats'
        with vu.file_reader(tmp_path, 'r') as f:
            ds_tmp = f[tmp_key]
            n_features = len(ds_tmp.attrs['feature_names'])

        # require the output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, dtype='float32', shape=(self.number_of_labels, n_features),
                              chunks=(chunk_size, 1), compression='gzip')

        # update the task config
        config.update({'output_path': self.output_path, 'output_key': self.output_key,
                       'tmp_path': tmp_path, 'tmp_key': tmp_key,
                       'node_chunk_size': chunk_size})

        node_block_list = vu.blocks_in_volume([self.number_of_labels], [chunk_size])

        n_jobs = min(len(node_block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, node_block_list, config, consecutive_blocks=True,
                          job_prefix=self.prefix)
        self.submit_jobs(n_jobs, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs, self.prefix)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.prefix))


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

def merge_feats(feat_name, this_feats, prev_feats,
                this_counts, prev_counts, tot_counts):
    assert len(this_feats) == len(prev_feats) == len(this_counts)
    if feat_name == 'count':
        return tot_counts
    elif feat_name == 'mean':
        return (this_counts * this_feats + prev_counts * prev_feats) / tot_counts
    elif feat_name == 'minimum':
        return np.minimum(this_feats, prev_feats)
    elif feat_name == 'maximum':
        return np.maximum(this_feats, prev_feats)
    else:
        raise ValueError("Invalid feature name %s" % feat_name)


def _extract_and_merge_region_features(blocking, ds_in, ds, node_begin, node_end, feature_names):
    fu.log("processing node range %i to %i" % (node_begin, node_end))
    n_nodes_chunk = node_end - node_begin
    n_features = len(feature_names)
    features = np.zeros((n_nodes_chunk, n_features), dtype='float32')

    chunks = ds_in.chunks
    for block_id in range(blocking.numberOfBlocks):
        block = blocking.getBlock(block_id)
        chunk_id = tuple(beg // ch for beg, ch in zip(block.begin, chunks))

        # load the data
        data = ds_in.read_chunk(chunk_id)
        if data is None:
            continue

        # extract the ids from the serialization
        n_cols = len(feature_names) + 1
        ids = data[::n_cols].astype('uint64')

        # check if any ids overlap with our id range
        overlap_mask = np.logical_and(ids >= node_begin,
                                      ids < node_end)
        if overlap_mask.sum() == 0:
            continue

        # extract the region features from the serialization
        feats = {}
        for feat_id, feat_name in enumerate(feature_names, 1):
            feats[feat_name] = data[feat_id::n_cols]

        # normalize the ids to the chunk
        overlapping_ids = ids[overlap_mask]
        overlapping_ids -= node_begin

        # compute the count features
        this_counts = feats['count'][overlap_mask]
        prev_counts = features[overlapping_ids, 0]
        assert len(this_counts) == len(prev_counts)
        tot_counts = prev_counts + this_counts

        # update all features
        for feat_id, feat_name in enumerate(feature_names):
            features[overlapping_ids, feat_id] = merge_feats(feat_name,
                                                             feats[feat_name][overlap_mask],
                                                             features[overlapping_ids, feat_id],
                                                             this_counts, prev_counts, tot_counts)

    features[np.isnan(features)] = 0.
    ds[node_begin:node_end, :] = features


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
        feature_names = ds_in.attrs['feature_names']
        assert feature_names[0] == 'count'

        ds = f[output_key]
        n_nodes = ds.shape[0]

        node_blocking = nt.blocking([0], [n_nodes], [node_chunk_size])
        node_begin = node_blocking.getBlock(node_block_list[0]).begin[0]
        node_end = node_blocking.getBlock(node_block_list[-1]).end[0]

        shape = list(ds_in.shape)
        chunks = list(ds_in.chunks)
        blocking = nt.blocking([0, 0, 0], shape, chunks)

        _extract_and_merge_region_features(blocking, ds_in, ds,
                                           node_begin, node_end, feature_names)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_region_features(job_id, path)
