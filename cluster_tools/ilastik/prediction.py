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


class PredictionBase(luigi.Task):
    """ Prediction base class
    """

    task_name = 'prediction'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    ilastik_folder = luigi.Parameter()
    ilastik_project = luigi.Parameter()
    halo = luigi.ListParameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter(default=None)
    n_channels = luigi.IntParameter(default=1)

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        shape = vu.get_shape(self.input_path, self.input_key)
        # FIXME we should be able to specify xyzc vs cyzx
        if len(shape) == 4:
            shape = shape[1:]
        assert len(shape) == 3
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'halo': self.halo,
                       'ilastik_project': self.ilastik_project,
                       'ilastik_folder': self.ilastik_folder,
                       'block_shape': block_shape, 'tmp_folder': self.tmp_folder})
        # if the output key is not None, we have a z5 file and
        # need to require the dataset
        if self.output_key is not None:
            config.update({'output_key': self.output_key})
            chunks = tuple(bs // 2 for bs in block_shape)
            if self.n_channels > 1:
                shape = (self.n_channels,) + shape
                chunks = (1,) + chunks
            with vu.file_reader(self.output_path) as f:
                f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                                  dtype='float32', compression='gzip')

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class PredictionLocal(PredictionBase, LocalTask):
    """ Prediction on local machine
    """
    pass


class PredictionSlurm(PredictionBase, SlurmTask):
    """ Prediction on slurm cluster
    """
    pass


class PredictionLSF(PredictionBase, LSFTask):
    """ Prediction on lsf cluster
    """
    pass


def _predict_block_impl(block_id, block, input_path, input_key,
                        output_prefix, ilastik_folder, ilastik_project):
    # assemble the ilastik headless command
    input_str = '%s/%s' % (input_path, input_key)
    output_str = '%s_block%i.h5' % (output_prefix, block_id)
    fu.log("Serializing block %i to %s" % (block_id, output_str))

    start, stop = block.begin, block.end
    # ilastik always wants 5d coordinates, for now we only support 3d
    assert len(start) == len(stop) == 3, "Only support 3d data"

    # FIXME is there a way to read axis order from ilastik
    # FIXME ilastik docu advice is to give tcxyz, but this does not work!!!
    # currently the only working option is xyzc
    # FIXME specify axis order properly
    start = [str(st) for st in start] + ['None']
    stop = [str(st) for st in stop] + ['None']

    # start = ['None'] + [str(st) for st in start]
    # stop = ['None'] + [str(st) for st in stop]

    start = '(%s)' % ','.join(start)
    stop = '(%s)' % ','.join(stop)

    subregion_str = '[%s, %s]' % (start, stop)
    fu.log("Subregion: %s" % subregion_str)

    ilastik_exe = os.path.join(ilastik_folder, 'run_ilastik.sh')
    assert os.path.exists(ilastik_exe), ilastik_exe
    cmd = [ilastik_exe, '--headless',
           '--project=%s' % ilastik_project,
           '--output_format=compressed hdf5',
           '--raw_data=%s' % input_str,
           '--cutout_subregion=%s' % subregion_str,
           '--output_filename_format=%s' % output_str,
           '--readonly=1']

    # log the cmd string
    cmd_str = ' '.join(cmd)
    fu.log("Calling ilastik with command %s" % cmd_str)

    # switch to the ilastik folder and call ilastik command
    try:
        # err = subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        # subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        fu.log("Call to ilastik failed")
        fu.log("with %s" % str(e.returncode))
        # fu.log("error message: %s" % e.output)
        raise e


def _predict_block(block_id, blocking,
                   input_path, input_key,
                   output_prefix, halo,
                   ilastik_folder, ilastik_project):
    fu.log("Start processing block %i" % block_id)
    block = blocking.getBlockWithHalo(block_id, halo).outerBlock
    _predict_block_impl(block_id, block, input_path, input_key,
                        output_prefix, ilastik_folder, ilastik_project)
    fu.log_block_success(block_id)


def _predict_and_serialize_block(block_id, blocking, input_path, input_key,
                                 output_prefix, halo,
                                 ilastik_folder, ilastik_project, ds):
    fu.log("Start processing block %i" % block_id)
    block = blocking.getBlockWithHalo(block_id, halo)
    _predict_block_impl(block_id, block.outerBlock, input_path, input_key,
                        output_prefix, ilastik_folder, ilastik_project)
    bb = vu.block_to_bb(block.innerBlock)
    inner_bb = vu.block_to_bb(block.innerBlockLocal)
    path = '%s_block%i.h5' % (output_prefix, block_id)

    with vu.file_reader(path, 'r') as f:
        pred = f['exported_data'][:].squeeze()
        assert pred.ndim in (3, 4), '%i' % pred.ndim

        if pred.ndim == 4:
            n_channels = ds.shape[0]
            bb = (slice(None),) + bb
            inner_bb = (slice(None),) + inner_bb
            # check if we need to transpose
            if pred.shape[-1] == n_channels:
                pred = pred.transpose((3, 0, 1, 2))
            else:
                assert pred.shape[0] == n_channels,\
                    "Expected first axis to be channel axis, but got shape %s" % str(pred.shape)

    pred = pred[inner_bb]
    ds[bb] = pred
    # os.remove(path)
    fu.log_block_success(block_id)


def prediction(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    halo = config['halo']
    ilastik_project = config['ilastik_project']
    ilastik_folder = config['ilastik_folder']
    tmp_folder = config['tmp_folder']

    n_threads = config.get('threads_per_job', 1)
    mem_in_gb = config.get('mem_limit', 1)
    mem_in_mb = mem_in_gb * 1000

    output_path = config['output_path']
    output_key = config.get('output_key', None)

    # set lazyflow environment variables
    os.environ['LAZYFLOW_THREADS'] = str(n_threads)
    os.environ['LAZYFLOW_TOTAL_RAM_MB'] = str(mem_in_mb)

    assert os.path.exists(ilastik_project), ilastik_project
    assert os.path.exists(ilastik_folder), ilastik_folder
    assert os.path.exists(input_path)

    block_shape = config['block_shape']
    block_list = config['block_list']

    shape = vu.get_shape(input_path, input_key)
    if len(shape) == 4:
        shape = shape[1:]

    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    output_prefix = os.path.join(tmp_folder, 'ilpred')
    if output_key is None:
        fu.log("predicting blocks with temporary serialization")
        for block_id in block_list:
            _predict_block(block_id, blocking, input_path, input_key,
                           output_prefix, halo,
                           ilastik_folder, ilastik_project)
    else:
        ds_out = vu.file_reader(output_path)[output_key]
        fu.log("predicting blocks and serializing to")
        fu.log("%s:%s" % (output_path, output_key))
        for block_id in block_list:
            _predict_and_serialize_block(block_id, blocking, input_path, input_key,
                                         output_prefix, halo,
                                         ilastik_folder, ilastik_project,
                                         ds_out)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    prediction(job_id, path)
