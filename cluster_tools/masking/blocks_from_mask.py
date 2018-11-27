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


# NOTE we don't exclude the ignore label here, but ignore
# it in the graph extraction already
class BlocksFromMaskBase(luigi.Task):
    """ BlocksFromMask base class
    """

    task_name = 'blocks_from_mask'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    mask_path = luigi.Parameter()
    mask_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        with vu.file_reader(self.input_path) as f:
            n_edges = f[self.input_key].shape[0]
        # chunk size = 64**3
        chunk_size = min(262144, n_edges)

        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=(n_edges,), compression='gzip',
                              dtype='float32', chunks=(chunk_size,))

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'features_path': self.features_path, 'features_key': self.features_key})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class BlocksFromMaskLocal(BlocksFromMaskBase, LocalTask):
    """ BlocksFromMask on local machine
    """
    pass


class BlocksFromMaskSlurm(BlocksFromMaskBase, SlurmTask):
    """ BlocksFromMask on slurm cluster
    """
    pass


class BlocksFromMaskLSF(BlocksFromMaskBase, LSFTask):
    """ BlocksFromMask on lsf cluster
    """
    pass


#
# Implementation
#

def blocks_from_mask(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    mask_path = config['mask_path']
    mask_key = config['mask_key']
    output_path = config['output_path']
    output_key = config['output_key']

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    blocks_from_mask(job_id, path)
