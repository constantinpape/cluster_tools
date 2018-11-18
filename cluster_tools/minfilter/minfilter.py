#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Watershed Tasks
#

# TODO implement minfilter of downsampled mask !
class MinfilterBase(luigi.Task):
    """ Watershed base class
    """

    task_name = 'minfilter'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'filter_shape': (10, 100, 100)})
        return config

    def clean_up_for_retry(self, block_list):
        # TODO does this work with the mixin pattern?
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        # load the watershed config
        # TODO check more parameters here
        config = self.get_task_config()

        # require output dataset
        # TODO read chunks from config
        chunks = tuple(bs // 2 for bs in block_shape)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, ws_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class MinfilterLocal(MinfilterBase, LocalTask):
    """ Minfilter on local machine
    """
    pass


class MinfilterSlurm(MinfilterBase, SlurmTask):
    """ Minfilter on slurm cluster
    """
    pass


class MinfilterLSF(MinfilterBase, LSFTask):
    """ Minfilter on lsf cluster
    """
    pass


#
# Implementation
#


def _minfilter_block(block_id, blocking, halo, ds_in, ds_out, filter_shape):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlockWithHalo(block_id, halo)
    outer_roi = vu.block_to_bb(block.outerBlock)
    inner_roi = vu.block_to_bb(block.innerBlock)
    local_roi = vu.block_to_bb(block.innerBlockLocal)
    mask = ds_in[outer_roi]
    min_filter_mask = minimum_filter(mask, size=filter_shape)
    ds_out[inner_roi] = min_filter_mask[local_roi]
    fu.log_block_success(block_id)


# TODO minfilter from downsampled mask
def minfilter(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # input/output files
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    # blocks and task config
    block_list = config['block_list']
    filter_shape = config['filter_shape']

    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]
        shape = ds_in.shape

        blocking = nt.blocking(roiBegin=[0, 0, 0],
                               roiEnd=list(shape),
                               blockShape=list(block_shape))

        # TODO is half of the halo really enough halo ?
        halo = list(fshape // 2 for fshape in filter_shape)
        [_minfilter_block(block_id, blocking, halo, ds_in,
                           ds_out, filter_shape) for block_id in block_list]
    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    watershed(job_id, path)
