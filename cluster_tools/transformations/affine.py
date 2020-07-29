#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt
from elf.transformation import parameters_to_matrix
from elf.wrapper.affine_volume import AffineVolume

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# affine transformation tasks
#

class AffineBase(luigi.Task):
    """ affine base class
    """

    task_name = 'affine'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    # transformation parameters
    transformation = luigi.ListParameter()
    shape = luigi.ListParameter()
    order = luigi.IntParameter(default=0)

    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'chunks': None, 'compression': 'gzip',
                       'fill_value': 0, 'sigma_anti_aliasing': None})
        return config

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def compute_shape(self, input_shape):
        pass

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            dtype = f[self.input_key].dtype

        # load the config
        task_config = self.get_task_config()
        compression = task_config.pop('compression', 'gzip')
        chunks = task_config.pop('chunks', None)
        if chunks is None:
            chunks = block_shape

        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=self.shape, chunks=tuple(chunks),
                              compression=compression, dtype=dtype)

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key,
                            'block_shape': block_shape, 'transformation': self.transformation,
                            'shape': self.shape, 'order': self.order})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(self.shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        self._write_log("scheduled %i blocks to run" % len(block_list))

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class AffineLocal(AffineBase, LocalTask):
    """
    Affine intensity transform on local machine
    """
    pass


class AffineSlurm(AffineBase, SlurmTask):
    """
    copy on slurm cluster
    Affine intensity transform on slurm cluster
    """
    pass


class AffineLSF(AffineBase, LSFTask):
    """
    Affine intensity transform on lsf cluster
    """
    pass


#
# Implementation
#


def _copy_blocks(ds_in, ds_out, blocking, block_list):
    for block_id in block_list:
        fu.log("start processing block %i" % block_id)
        block = blocking.getBlock(block_id)
        bb = vu.block_to_bb(block)
        data = ds_in[bb]
        ds_out[bb] = data


def affine(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input and output cofig
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    # compute the transformation matrix
    trafo = config['transformation']
    trafo = parameters_to_matrix(trafo)

    # load additional trafo parameter
    order = config['order']
    fill_value = config['fill_value']
    sigma_anti_aliasing = config['sigma_anti_aliasing']

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]
        shape = ds_out.shape

        # wrap the input dataset into the affine volume wrapper
        ds_in = AffineVolume(ds_in, shape=shape, affine_matrix=trafo,
                             order=order, fill_value=fill_value,
                             sigma_anti_aliasing=sigma_anti_aliasing)

        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)
        _copy_blocks(ds_in, ds_out, blocking, block_list)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    affine(job_id, path)
