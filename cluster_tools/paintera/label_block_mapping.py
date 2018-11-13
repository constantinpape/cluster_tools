#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class LabelBlockMappingBase(luigi.Task):
    """ LabelBlockMapping base class
    """

    task_name = 'label_block_mapping'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    number_of_labels = luigi.IntParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, _, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # TODO support roi
        assert roi_begin is None and roi_end is None

        # shape and chunks for the id space
        ds_shape = (int(2**63 - 1),) # open-ended shape
        actual_shape = (self.number_of_labels,) # actual shape needed for blocking
        # we use a chunk-size of 10k, but this could also be a parameter
        chunks = (10000,)
        # we use the chunks as block-shape
        block_shape = chunks

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(actual_shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        # create the output dataset
        with vu.file_reader(self.output_path) as f:
            compression = 'gzip'
            f.require_dataset(self.output_key, shape=ds_shape, compression=compression,
                              chunks=chunks, dtype='int8')
        n_jobs = min(len(block_list), self.max_jobs)

        # we don't need any additional config besides the paths
        config = {"input_path": self.input_path, "input_key": self.input_key,
                  "output_path": self.output_path, "output_key": self.output_key,
                  "block_shape": block_shape, "number_of_labels": self.number_of_labels}
        self._write_log('scheduling %i blocks to be processed' % len(block_list))

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class LabelBlockMappingLocal(LabelBlockMappingBase, LocalTask):
    """
    LabelBlockMapping on local machine
    """
    pass


class LabelBlockMappingSlurm(LabelBlockMappingBase, SlurmTask):
    """
    LabelBlockMapping on slurm cluster
    """
    pass


class LabelBlockMappingLSF(LabelBlockMappingBase, LSFTask):
    """
    LabelBlockMapping on lsf cluster
    """
    pass


#
# Implementation
#


def _label_to_block_mapping(input_path, input_key,
                            output_path, output_key,
                            blocking, block_list):
    for block_id in block_list:
        id_space_block = blocking.getBlock(block_id)
        id_start, id_stop = id_space_block.begin[0], id_space_block.end[0]
        ndist.serializeBlockMapping(os.path.join(input_path, input_key),
                                    os.path.join(output_path, output_key),
                                    id_start, id_stop)
        fu.log_block_success(block_id)


def label_block_mapping(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    number_of_labels = config['number_of_labels']

    block_list = config['block_list']
    block_shape = config['block_shape']

    # read id spacei chunks from output; make id space blocking
    with vu.file_reader(output_path) as f:
        chunks = list(f[output_key].chunks)
    # shape for max ids
    shape = [number_of_labels]
    blocking = nt.blocking([0], shape, chunks)
    _label_to_block_mapping(input_path, input_key,
                            output_path, output_key,
                            blocking, block_list)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    label_block_mapping(job_id, path)
