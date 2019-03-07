#! /bin/python

import os
import sys
import json
import numpy as np

import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#


class FilterBlocksBase(luigi.Task):
    """ FilterBlocks base class
    """

    task_name = 'filter_blocks'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    filter_path = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'block_shape': block_shape, 'filter_path': self.filter_path,
                       'output_path': self.output_path, 'output_key': self.output_key})

        # create output dataset
        shape = vu.get_shape(self.input_path, self.input_key)
        chunks = tuple(min(bs // 2, sh) for bs, sh in zip(block_shape, shape))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape,
                              dtype='uint64', chunks=chunks,
                              compression='gzip')

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape,
                                             roi_begin, roi_end)
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


class FilterBlocksLocal(FilterBlocksBase, LocalTask):
    """ FilterBlocks on local machine
    """
    pass


class FilterBlocksSlurm(FilterBlocksBase, SlurmTask):
    """ FilterBlocks on slurm cluster
    """
    pass


class FilterBlocksLSF(FilterBlocksBase, LSFTask):
    """ FilterBlocks on lsf cluster
    """
    pass


#
# Implementation
#

def _filter_block(blocking, block_id,
                  ds_in, ds_out, filter_ids):
    fu.log("start processing block %i" % block_id)
    # read labels and input in this block
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    seg = ds_in[bb]

    # check if segmentation block is empty
    if seg.sum() == 0:
        fu.log_block_success(block_id)
        return

    # check for filter_ids
    filter_mask = np.in1d(seg, filter_ids).reshape(seg.shape)
    seg[filter_mask] = 0
    ds_out[bb] = seg
    fu.log_block_success(block_id)


def _filter_block_inplace(blocking, block_id,
                          ds, filter_ids):
    fu.log("start processing block %i" % block_id)
    # read labels and input in this block
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    seg = ds[bb]

    # check if segmentation block is empty
    if seg.sum() == 0:
        fu.log_block_success(block_id)
        return

    # check for filter_ids
    filter_mask = np.in1d(seg, filter_ids).reshape(seg.shape)

    # check if we filter any ids
    if filter_mask.sum() == 0:
        fu.log_block_success(block_id)
        return

    seg[filter_mask] = 0
    ds[bb] = seg
    fu.log_block_success(block_id)


def filter_blocks(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    filter_path = config['filter_path']

    block_shape = config['block_shape']
    block_list = config['block_list']

    with vu.file_reader(input_path, 'r') as f:
        shape = f[input_key].shape

    blocking = nt.blocking([0, 0, 0],
                           list(shape),
                           list(block_shape))

    with open(filter_path) as f:
        filter_ids = json.load(f)
    filter_ids = np.array(filter_ids, dtype='uint64')

    in_place = input_path == output_path and input_key == output_key
    if in_place:
        fu.log("filtering blocks in place")
        with vu.file_reader(input_path) as f:
            ds = f[input_key]
            for block_id in block_list:
                _filter_block_inplace(blocking, block_id,
                                      ds, filter_ids)
    else:
        fu.log("filtering blocks to new dataset")
        with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:

            ds_in = f_in[input_key]
            ds_out = f_out[output_key]

            for block_id in block_list:
                _filter_block(blocking, block_id,
                              ds_in, ds_out, filter_ids)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    filter_blocks(job_id, path)
