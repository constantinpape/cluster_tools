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


class MergePredictionsBase(luigi.Task):
    """ MergePredictions base class
    """

    task_name = 'merge_predictions'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    tmp_prefix = luigi.Parameter()
    halo = luigi.ListParameter()
    n_channels = luigi.IntParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        shape = vu.get_shape(self.input_path, self.input_key)
        chunks = tuple(bs // 2 for bs in block_shape)

        if self.n_channels > 1:
            shape = (self.n_channels,) + shape
            chunks = (1,) + chunks

        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, compression='gzip',
                              dtype='float32', chunks=chunks)

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'tmp_prefix': self.tmp_prefix, 'halo': self.halo,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape, 'n_channels': self.n_channels})

        # FIXME only support 1 job for now, because this is tailored to hdf5
        n_jobs = 1
        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class MergePredictionsLocal(MergePredictionsBase, LocalTask):
    """ MergePredictions on local machine
    """
    pass


class MergePredictionsSlurm(MergePredictionsBase, SlurmTask):
    """ MergePredictions on slurm cluster
    """
    pass


class MergePredictionsLSF(MergePredictionsBase, LSFTask):
    """ MergePredictions on lsf cluster
    """
    pass


def _merge_block(block_id, blocking, ds, tmp_prefix, halo, n_channels):
    fu.log("Start processing block %i" % block_id)
    block = blocking.getBlockWithHalo(block_id, halo)

    inner_bb = vu.block_to_bb(block.innerBlock)
    local_bb = vu.block_to_bb(block.innerBlockLocal)

    # TODO does ilastik output 5D ????
    # Note that ilastik uses zyxc, while we use czyx
    if n_channels > 1:
        inner_bb = (slice(None),) + inner_bb
        local_bb = local_bb + (slice(None),)

    tmp_path = '%s_block%i.h5' % (tmp_prefix, block_id)
    with vu.file_reader(tmp_path, 'r') as f:
        data = f['exported_data'][local_bb]
    if n_channels > 1:
        data = data.transpose((3, 0, 1, 2))

    ds[inner_bb] = data
    # TODO remove tmp files once we are sure all works
    # os.remove(tmp_path)
    fu.log_block_success(block_id)


def merge_predictions(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    output_path = config['output_path']
    output_key = config['output_key']
    tmp_prefix = config['tmp_prefix']
    halo = config['halo']
    n_channels = config['n_channels']

    shape = vu.get_shape(output_path, output_key)
    if len(shape) > 3:
        shape = shape[-3:]
    block_shape = config['block_shape']
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # TODO we could parallelize this
    with vu.file_reader(output_path) as f:
        ds = f[output_key]
        for block_id in range(blocking.numberOfBlocks):
            _merge_block(block_id, blocking, ds, tmp_prefix, halo,
                         n_channels)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_predictions(job_id, path)
