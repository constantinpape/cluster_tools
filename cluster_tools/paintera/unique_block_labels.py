#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class UniqueBlockLabelsBase(luigi.Task):
    """ UniqueBlockLabels base class
    """

    task_name = 'unique_block_labels'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, _, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # read shape chunks and mulit-set from input
        with vu.file_reader(self.input_path) as f:
            ds = f[self.input_key]
            shape = ds.shape
            chunks = ds.chunks
            dtype = ds.dtype
            is_multiset = ds.attrs.get('isLabelMultiset', False)
        # TODO implement label multiset
        if is_multiset:
            raise NotImplementedError("Multiset is not supported yet")

        # we use the chunks as block-shape
        block_shape = chunks

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        # create the output dataset
        with vu.file_reader(self.output_path) as f:
            # TODO gzip compression might be problematic ...
            compression = 'raw'
            f.require_dataset(self.output_key, shape=shape, compression=compression,
                              chunks=chunks, dtype=dtype)
        n_jobs = min(len(block_list), self.max_jobs)

        # we don't need any additional config besides the paths
        config = {"input_path": self.input_path, "input_key": self.input_key,
                  "output_path": self.output_path, "output_key": self.output_key,
                  "block_shape": block_shape, "is_multiset": is_multiset}
        self._write_log('scheduling %i blocks to be processed' % len(block_list))

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class UniqueBlockLabelsLocal(UniqueBlockLabelsBase, LocalTask):
    """
    UniqueBlockLabels on local machine
    """
    pass


class UniqueBlockLabelsSlurm(UniqueBlockLabelsBase, SlurmTask):
    """
    UniqueBlockLabels on slurm cluster
    """
    pass


class UniqueBlockLabelsLSF(UniqueBlockLabelsBase, LSFTask):
    """
    UniqueBlockLabels on lsf cluster
    """
    pass


#
# Implementation
#


def _uniques_default(ds, ds_out, blocking, block_list):
    for block_id in block_list:
        block_coord = blocking.getBlock(block_id).begin
        chunk_id = tuple(bl // ch for bl, ch in zip(block_coord, ds.chunks))

        labels = ds.read_chunk(chunk_id)
        uniques = np.unique(labels)
        # TODO can we skip blocks with only 0 as  label in paintera format ??
        if len(uniques) == 1 and uniques[0] == 0:
            return
        ds_out.write_chunk(chunk_id, uniques, True)
        fu.log_block_success(block_id)


# TODO implement properly
def _uniques_multiset(ds, ds_out, blocking, block_list):
    for block_id in block_list:
        block_coord = blocking.getBlock(block_id).begin
        chunk_id = tuple(bl // ch for bl, ch in zip(block_coord, ds.chunks))


def unique_block_labels(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    block_list = config['block_list']
    block_shape = config['block_shape']
    is_multiset = config['is_multiset']

    # open the input file
    with vu.file_reader(input_path, 'r') as f, vu.file_reader(output_path) as f_out:
        ds = f[input_key]
        ds_out = f_out[output_key]
        chunks = ds.chunks
        shape = ds.shape
        assert tuple(chunks) == tuple(block_shape),\
            "Chunks %s and block shape %s must agree" % (str(chunks), str(block_shape))

        blocking = nt.blocking([0, 0, 0], shape, block_shape)

        if is_multiset:
            _uniques_multiset(ds, ds_out, blocking, block_list)
        else:
            _uniques_default(ds, ds_out, blocking, block_list)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    unique_block_labels(job_id, path)
