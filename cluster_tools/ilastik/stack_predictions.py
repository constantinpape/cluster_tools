#! /usr/bin/python

import os
import sys
import argparse
import pickle
import json
import subprocess

import numpy as np
import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class StackPredictionsBase(luigi.Task):
    """ StackPredictions base class
    """

    task_name = 'stack_predictions'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    raw_path = luigi.Parameter()
    raw_key = luigi.Parameter()
    pred_path = luigi.Parameter()
    pred_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dtype = luigi.Parameter(default='float32')

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # determine the output shape
        raw_shape = vu.get_shape(self.raw_path, self.raw_key)
        with vu.file_reader(self.pred_path, 'r') as f:
            ds_pred = f[self.pred_key]
            pred_shape = ds_pred.shape
            # TODO make chunks a parameter
            chunks = ds_pred.chunks

        assert len(pred_shape) == 4
        assert len(raw_shape) == 3
        assert pred_shape[1:] == raw_shape
        out_shape = (1 + pred_shape[0],) + raw_shape
        block_list = vu.blocks_in_volume(raw_shape, block_shape, roi_begin, roi_end)

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'raw_path': self.raw_path, 'raw_key': self.raw_key,
                       'pred_path': self.pred_path, 'pred_key': self.pred_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape})

        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=out_shape, chunks=chunks,
                              dtype=self.dtype, compression='gzip')

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class StackPredictionsLocal(StackPredictionsBase, LocalTask):
    """ StackPredictions on local machine
    """
    pass


class StackPredictionsSlurm(StackPredictionsBase, SlurmTask):
    """ StackPredictions on slurm cluster
    """
    pass


class StackPredictionsLSF(StackPredictionsBase, LSFTask):
    """ StackPredictions on lsf cluster
    """
    pass


def cast(input_, dtype):
    if np.dtype(input_.dtype) == np.dtype(dtype):
        return input_
    # TODO allow more general casting
    assert dtype in ('float32', 'uint8')
    # option one: cast float to uint8
    if dtype == 'uint8':
        input_ *= 255
        return input_.astype('uint8')
    # option two: cast uint8 to float
    else:
        input_ = input_.astype('float32')
        input_ /= input_.max()
        return input_


def stack_block(block_id, blocking, ds_raw, ds_pred, ds_out, dtype):
    fu.log("start processing block %i" % block_id)
    bb = vu.block_to_bb(blocking.getBlock(block_id))
    raw = cast(ds_raw[bb], dtype)
    bb = (slice(None),) + bb
    pred = cast(ds_pred[bb], dtype)
    out = np.concatenate([raw[None], pred], axis=0)
    ds_out[bb] = out
    fu.log_block_success(block_id)


def stack_predictions(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    raw_path = config['raw_path']
    raw_key = config['raw_key']
    pred_path = config['pred_path']
    pred_key = config['pred_key']

    output_path = config['output_path']
    output_key = config['output_key']

    block_shape = config['block_shape']
    block_list = config['block_list']

    with vu.file_reader(raw_path, 'r') as fr,\
        vu.file_reader(pred_path, 'r') as fp,\
        vu.file_reader(output_path) as fout:

        ds_raw = fr[raw_key]
        ds_pred = fp[pred_key]
        ds_out = fout[output_key]

        dtype = str(ds_out.dtype)

        shape = ds_raw.shape
        blocking = nt.blocking([0, 0, 0], shape, block_shape)

        for block_id in block_list:
            stack_block(block_id, blocking, ds_raw, ds_pred, ds_out, dtype)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    stack_predictions(job_id, path)
