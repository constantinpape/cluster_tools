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
    output_prefix = luigi.Parameter()
    ilastik_folder = luigi.Parameter()
    ilastik_project = luigi.Parameter()
    halo = luigi.ListParameter()

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        shape = vu.get_shape(self.input_path, self.input_key)

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_prefix': self.output_prefix, 'halo': self.halo,
                       'ilastik_project': self.ilastik_project,
                       'ilastik_folder': self.ilastik_folder,
                       'block_shape': block_shape})

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)

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


def _predict_block(block_id, blocking, input_path, input_key,
                   output_prefix, halo,
                   ilastik_folder, ilastik_project):
    fu.log("Start processing block %i" % block_id)
    block = blocking.getBlockWithHalo(block_id, halo).outerBlock

    # assemble the ilastik headless command
    input_str = '%s/%s' % (input_path, input_key)
    subregion_str = ''
    output_str = '%s_block%i.h5' % (output_prefix, block_id)

    start, stop = block.begin, block.end
    # ilastik always wants 5d coordinates, for now we only support 3d
    assert len(start) == len(stop) == 3, "Only support 3d data"
    # FIXME ilastik docu advice is to give tcxyz, but this does not work!!!
    # currently the only working option is xyzc
    start = [str(st) for st in start] + ['None']
    stop = [str(st) for st in stop] + ['None']
    start = '(%s)' % ','.join(start)
    stop = '(%s)' % ','.join(stop)
    subregion_str = '[%s, %s]' % (start, stop)
    fu.log("Subregion: %s" % subregion_str)

    cmd = ['./run_ilastik.sh', '--headless',
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
    cwd = os.getcwd()
    os.chdir(ilastik_folder)
    try:
        # err = subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        # subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        fu.log("Call to ilastik failed")
        fu.log("with %s" % str(e.returncode))
        # fu.log("error message: %s" % e.output)
        raise e
    os.chdir(cwd)

    fu.log_block_success(block_id)


def prediction(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_prefix = config['output_prefix']
    halo = config['halo']
    ilastik_project = config['ilastik_project']
    ilastik_folder = config['ilastik_folder']
    n_threads = config.get('threads_per_job', 1)
    mem_in_gb = config.get('mem_limit', 1)
    mem_in_mb = mem_in_gb * 1000

    # set lazyflow environment variables
    os.environ['LAZYFLOW_THREADS'] = str(n_threads)
    os.environ['LAZYFLOW_TOTAL_RAM_MB'] = str(mem_in_mb)

    assert os.path.exists(ilastik_project), ilastik_project
    assert os.path.exists(ilastik_folder), ilastik_folder
    assert os.path.exists(input_path)

    block_shape = config['block_shape']
    block_list = config['block_list']

    shape = vu.get_shape(input_path, input_key)

    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    for block_id in block_list:
        _predict_block(block_id, blocking, input_path, input_key,
                       output_prefix, halo,
                       ilastik_folder, ilastik_project)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    prediction(job_id, path)
