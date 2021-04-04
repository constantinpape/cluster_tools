#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Block-wise gradient computation tasks
#

class ToBoundariesBase(luigi.Task):
    """ ToBoundaries base class
    """
    task_name = 'to_boundaries'
    src_file = os.path.abspath(__file__)
    allow_retry = True

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    accumulation_method = luigi.Parameter(default='mean')
    channel_begin = luigi.IntParameter(default=0)
    channel_end = luigi.IntParameter(default=None)
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, _, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape
            dtype = f[self.input_key].dtype
            chunks = f[self.input_key].chunks[1:]

        assert self.channel_begin < shape[0]
        channel_end = shape[0] if self.channel_end is None else self.channel_end
        assert channel_end <= shape[0]

        shape = shape[1:]
        assert len(shape) == 3
        block_shape = chunks

        config = self.get_task_config()
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape, 'accumulation_method': self.accumulation_method,
                       'channel_begin': self.channel_begin, 'channel_end': channel_end})

        # make output dataset
        with vu.file_reader(self.output_path, 'a') as f:
            f.require_dataset(self.output_key, shape=shape, dtype=dtype,
                              compression='gzip', chunks=chunks)

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class ToBoundariesLocal(ToBoundariesBase, LocalTask):
    """
    ToBoundaries on local machine
    """
    pass


class ToBoundariesSlurm(ToBoundariesBase, SlurmTask):
    """
    ToBoundaries on slurm cluster
    """
    pass


class ToBoundariesLSF(ToBoundariesBase, LSFTask):
    """
    ToBoundaries on lsf cluster
    """
    pass


#
# Implementation
#


def _to_boundaries_block(block_id, blocking,
                         ds_in, ds_out, accumulator,
                         channel_begin, channel_end):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    bb_in = (slice(channel_begin, channel_end),) + bb
    affs = ds_in[bb_in]
    bd = accumulator(affs, axis=0).astype(ds_out.dtype)
    ds_out[bb] = bd
    fu.log_block_success(block_id)


def to_boundaries(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    block_list = config['block_list']
    block_shape = config['block_shape']

    accumulation_method = config['accumulation_method']
    channel_begin = config['channel_begin']
    channel_end = config['channel_end']

    accumulator = getattr(np, accumulation_method)

    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path, 'a') as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]
        shape = ds_out.shape
        blocking = nt.blocking([0, 0, 0], shape, block_shape)
        [_to_boundaries_block(block_id, blocking, ds_in, ds_out,
                              accumulator, channel_begin, channel_end)
         for block_id in block_list]
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    to_boundaries(job_id, path)
