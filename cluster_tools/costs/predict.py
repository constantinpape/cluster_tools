#! /usr/bin/python

import os
import sys
import argparse
import pickle
import json

import numpy as np
import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


# TODO enable retry with consecutive edges
class PredictBase(luigi.Task):
    """ Predict base class
    """

    task_name = 'predict'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    rf_path = luigi.Parameter()
    features_path = luigi.Parameter()
    features_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        # TODO does this work with the mixin pattern?
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        with vu.file_reader(self.features_path) as f:
            feat_shape = f[self.features_key].shape
        n_edges = feat_shape[0]
        # chunk size = 64**3
        chunk_size = min(262144, n_edges)

        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=(n_edges,), compression='gzip',
                              dtype='float32', chunks=(chunk_size,))

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'rf_path': self.rf_path,
                       'features_path': self.features_path, 'features_key': self.features_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'chunk_size': chunk_size, 'n_edges': n_edges})

        if self.n_retries == 0:
            edge_block_list = vu.blocks_in_volume([n_edges], [chunk_size])
        else:
            edge_block_list = self.block_list
            self.clean_up_for_retry(edge_block_list)

        n_jobs = min(len(edge_block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, edge_block_list, config,
                          consecutive_blocks=True)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class PredictLocal(PredictBase, LocalTask):
    """ Predict on local machine
    """
    pass


class PredictSlurm(PredictBase, SlurmTask):
    """ Predict on slurm cluster
    """
    pass


class PredictLSF(PredictBase, LSFTask):
    """ Predict on lsf cluster
    """
    pass


def predict(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    rf_path = config['rf_path']
    n_threads = config['threads_per_job']
    features_path = config['features_path']
    features_key = config['features_key']
    output_path = config['output_path']
    output_key = config['output_key']
    n_edges = config['n_edges']
    edge_chunk_size = config['chuk_size']
    edge_block_list = config['block_list']

    # assert that the edge block list is consecutive
    diff_list = np.diff(edge_block_list)
    assert (diff_list == 1).all()

    with open(rf_path, 'rb') as f:
        rf = pickle.load(f)
    rf.n_jobs = n_threads

    edge_blocking = nt.blocking([0], [n_edges], [edge_chunk_size])
    edge_begin = edge_blocking.getBlock(edge_block_list[0]).begin[0]
    edge_end = edge_blocking.getBlock(edge_block_list[-1]).end[0]

    feat_roi = np.s_[edge_begin:edge_end, :]
    with vu.file_reader(features_path) as f:
        ds = f[features_key]
        ds.n_threads = n_threads
        feats = ds[feat_roi]

    probs = rf.predict_proba(feats)[:, 1].astype('float32')
    with vu.file_reader(output_path) as f:
        ds = f[output_key]
        ds.n_threads = n_threads
        ds[edge_begin:edge_end] = probs

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    predict(job_id, path)
