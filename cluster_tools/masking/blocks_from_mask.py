#! /usr/bin/python

import os
import sys
import argparse
import pickle
import json
from concurrent import futures

import numpy as np
import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


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
    shape = luigi.ListParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, _, _ = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'mask_path': self.mask_path, 'mask_key': self.mask_key,
                       'output_path': self.output_path, 'shape': self.shape})

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


def _get_blocks_in_mask(mask, blocking, n_threads):

    def check_block(block_id):
        block = blocking.getBlock(block_id)
        bb = vu.block_to_bb(block)
        mask_bb = mask[bb]
        return block_id if np.sum(mask_bb) > 0 else None

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(check_block, block_id)
                 for block_id in range(blocking.numberOfBlocks)]
        blocks_in_mask = [t.result() for t in tasks]
    blocks_in_mask = [block_id for block_id
                      in blocks_in_mask if block_id is not None]
    return blocks_in_mask


def blocks_from_mask(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    mask_path = config['mask_path']
    mask_key = config['mask_key']
    output_path = config['output_path']
    shape = config['shape']
    block_shape = config['block_shape']
    n_threads = config.get('threads_per_job', 1)

    # NOTE we assume that the mask is small and will fit into memory
    with vu.file_reader(mask_path, 'r') as f:
        ds = f[mask_key]
        ds.n_threads = n_threads
        mask_data = ds[:]
    mask = vu.InterpolatedVolume(mask_data, tuple(shape))

    blocking = nt.blocking([0, 0, 0], shape, list(block_shape))
    blocks_in_mask = _get_blocks_in_mask(mask, blocking, n_threads)

    with open(output_path, 'w') as f:
        json.dump(blocks_in_mask, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    blocks_from_mask(job_id, path)
