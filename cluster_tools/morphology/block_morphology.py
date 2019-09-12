#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Morphology Tasks
#

class BlockMorphologyBase(luigi.Task):
    """ BlockMorphology base class
    """

    task_name = 'block_morphology'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    prefix = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'block_shape': block_shape,
                       'output_path': self.output_path,
                       'output_key': self.output_key})

        # create output dataset
        shape = vu.get_shape(self.input_path, self.input_key)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape,
                              dtype='float64',
                              chunks=tuple(block_shape),
                              compression='gzip')

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape,
                                             roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config, self.prefix)
        self.submit_jobs(n_jobs, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs, self.prefix)

    # part of the luigi API
    def output(self):
        outp = os.path.join(self.tmp_folder,
                            "%s_%s.log" % (self.task_name, self.prefix))
        return luigi.LocalTarget(outp)


class BlockMorphologyLocal(BlockMorphologyBase, LocalTask):
    """ BlockMorphology on local machine
    """
    pass


class BlockMorphologySlurm(BlockMorphologyBase, SlurmTask):
    """ BlockMorphology on slurm cluster
    """
    pass


class BlockMorphologyLSF(BlockMorphologyBase, LSFTask):
    """ BlockMorphology on lsf cluster
    """
    pass


#
# Implementation
#

def _morphology_for_block(block_id, blocking, ds_in,
                          output_path, output_key):
    fu.log("start processing block %i" % block_id)
    # read labels and input in this block
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    seg = ds_in[bb]

    # check if segmentation block is empty
    if seg.sum() == 0:
        fu.log("block %i is empty" % block_id)
        fu.log_block_success(block_id)
        return

    chunk_id = tuple(beg // ch
                     for beg, ch in zip(block.begin,
                                        blocking.blockShape))
    # extract and save simple morphology:
    # - size of segments
    # - center of mass of segments
    # - minimum coordinates of segments
    # - maximum coordinates of segments
    ndist.computeAndSerializeMorphology(seg, block.begin,
                                        output_path, output_key,
                                        chunk_id)
    fu.log_block_success(block_id)


def block_morphology(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    block_shape = config['block_shape']
    block_list = config['block_list']

    with vu.file_reader(input_path, 'r') as f:
        shape = f[input_key].shape

    blocking = nt.blocking([0, 0, 0],
                           list(shape),
                           list(block_shape))

    with vu.file_reader(input_path, 'r') as f_in:
        ds_in = f_in[input_key]
        [_morphology_for_block(block_id, blocking, ds_in,
                               output_path, output_key)
         for block_id in block_list]
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    block_morphology(job_id, path)
