#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
from cluster_tools.utils.numpy_utils import set_numpy_threads
set_numpy_threads(1)
from elf.label_multiset import create_multiset_from_labels, serialize_multiset


class CreateMultisetBase(luigi.Task):
    """ CreateMultiset base class
    """

    task_name = 'create_multiset'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # dependency
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        config.update({'compression': 'gzip'})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        # load the create_multiset config
        config = self.get_task_config()

        compression = config.get('compression', 'gzip')
        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=tuple(block_shape),
                              compression=compression, dtype='uint8')

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape})
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        self._write_log('scheduling %i blocks to be processed' % len(block_list))
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class CreateMultisetLocal(CreateMultisetBase, LocalTask):
    """
    CreateMultiset on local machine
    """
    pass


class CreateMultisetSlurm(CreateMultisetBase, SlurmTask):
    """
    CreateMultiset on slurm cluster
    """
    pass


class CreateMultisetLSF(CreateMultisetBase, LSFTask):
    """
    CreateMultiset on lsf cluster
    """
    pass


#
# Implementation
#


def _create_multiset_block(blocking, block_id, ds_in, ds_out):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    labels = ds_in[bb]

    # # TODO I don't know if paintera supports this
    # if labels.sum() == 0:
    #     fu.log_block_success(block_id)
    #     return

    # compute multiset from input labels
    multiset = create_multiset_from_labels(labels)
    ser = serialize_multiset(multiset)

    chunk_id = tuple(bs // ch for bs, ch in zip(block.begin, ds_out.chunks))
    ds_out.write_chunk(chunk_id, ser, True)
    fu.log_block_success(block_id)


def write_metadata(ds_out, max_id):
    attrs = ds_out.attrs
    attrs['maxId'] = max_id
    attrs['isLabelMultiset'] = True


def create_multiset(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    # read the output config
    output_path = config['output_path']
    output_key = config['output_key']
    shape = list(vu.get_shape(output_path, output_key))

    # get the blocking
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        for block_id in block_list:
            _create_multiset_block(blocking, block_id, ds_in, ds_out)

        if job_id == 0:
            max_id = ds_in.attrs['maxId']
            write_metadata(ds_out, max_id)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    create_multiset(job_id, path)
