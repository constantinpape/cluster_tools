#! /usr/bin/python

import os
import sys
import argparse
import json

import numpy as np
import luigi
import z5py
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


# TODO support multi-channel filter
class ImageFilterBase(luigi.Task):
    """ Image filter base class
    """

    task_name = 'image_filter'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    filter_name = luigi.Parameter()
    sigma = luigi.Parameter()
    halo = luigi.ListParameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'apply_in_2d': False})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        config = self.get_task_config()
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'filter_name': self.filter_name, 'sigma': self.sigma,
                       'halo': self.halo, 'block_shape': block_shape})

        shape = vu.get_shape(self.input_path, self.input_key)
        chunks = tuple(min(bs // 2, sh) for bs, sh in zip(block_shape, shape))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, dtype='float32',
                              compression='gzip', chunks=chunks)

        if self.n_retries == 0:
            # get shape and make block config
            shape = vu.get_shape(self.input_path, self.input_key)
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
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


class ImageFilterLocal(ImageFilterBase, LocalTask):
    """ ImageFilter on local machine
    """
    pass


class ImageFilterSlurm(ImageFilterBase, SlurmTask):
    """ ImageFilter on slurm cluster
    """
    pass


class ImageFilterLSF(ImageFilterBase, LSFTask):
    """ ImageFilter on lsf cluster
    """
    pass


#
# Implementation
#


def _apply_filter(blocking, block_id, ds_in, ds_out,
                  halo, filter_name, sigma, apply_in_2d):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlockWithHalo(block_id, halo)
    bb_in = vu.block_to_bb(block.outerBlock)
    input_ = vu.normalize(ds_in[bb_in])
    response = vu.apply_filter(input_, filter_name, sigma, apply_in_2d)
    bb_out = vu.block_to_bb(block.innerBlock)
    inner_bb = vu.block_to_bb(block.innerBlockLocal)
    ds_out[bb_out] = response[inner_bb]
    fu.log_block_success(block_id)


def image_filter(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    block_list = config['block_list']
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    block_shape = config['block_shape']

    sigma = config['sigma']
    filter_name = config['filter_name']
    halo = config['halo']
    apply_in_2d = config.get('apply_in_2d', False)

    # iterate over blocks and apply filter
    with vu.file_reader(input_path, 'r') as f_in,\
        vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        shape = list(ds_in.shape)
        blocking = nt.blocking([0, 0, 0], shape, block_shape)

        for block_id in block_list:
            _apply_filter(blocking, block_id, ds_in, ds_out,
                          halo, filter_name, sigma, apply_in_2d)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    image_filter(job_id, path)
