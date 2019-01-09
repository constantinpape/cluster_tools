#! /usr/bin/python

import os
import sys
import json
import pickle

import luigi
import numpy as np
import nifty.tools as nt
from skimage.morphology import label

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Block-wise connected components tasks
#

class BlockComponentsBase(luigi.Task):
    """ BlockComponents base class
    """

    task_name = 'block_components'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()
    threshold = luigi.FloatParameter()
    threshold_mode = luigi.Parameter(default='greater')

    threshold_modes = ('greater', 'less', 'equal')

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        assert self.threshold_mode in self.threshold_modes
        config = self.get_task_config()
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'block_shape': block_shape,
                       'tmp_folder': self.tmp_folder,
                       'threshold': self.threshold,
                       'threshold_mode': self.threshold_mode})
        # make output dataset
        chunks = config.pop('chunks', None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)
        compression = config.pop('compression', 'gzip')
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key,  shape=shape, dtype='uint64',
                              compression=compression, chunks=chunks)

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)

        n_jobs = min(len(block_list), self.max_jobs)

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class BlockComponentsLocal(BlockComponentsBase, LocalTask):
    """
    BlockComponents on local machine
    """
    pass


class BlockComponentsSlurm(BlockComponentsBase, SlurmTask):
    """
    BlockComponents on slurm cluster
    """
    pass


class BlockComponentsLSF(BlockComponentsBase, LSFTask):
    """
    BlockComponents on lsf cluster
    """
    pass


def _cc_block(block_id, blocking,
              ds_in, ds_out, threshold,
              threshold_mode):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    input_ = ds_in[bb]

    if threshold_mode == 'greater':
        input_ = input_ > threshold
    elif threshold_mode == 'less':
        input_ = input_ < threshold
    elif threshold_mode == 'equal':
        input_ = input_ == threshold
    else:
        raise RuntimeError("Thresholding Mode %s not supported" % threshold_mode)

    if np.sum(input_) == 0:
        fu.log_block_success(block_id)
        return 0

    components = label(input_)
    ds_out[bb] = components
    fu.log_block_success(block_id)
    return int(components.max()) + 1


def block_components(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    block_list = config['block_list']
    tmp_folder = config['tmp_folder']
    block_shape = config['block_shape']
    threshold = config['threshold']
    threshold_mode = config['threshold_mode']

    fu.log("Applying threshold %f with mode %s" % (threshold, threshold_mode))

    with vu.file_reader(input_path, 'r') as f_in,\
        vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        shape = ds_in.shape
        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)

        offsets = [_cc_block(block_id, blocking,
                             ds_in, ds_out, threshold,
                             threshold_mode) for block_id in block_list]

    offset_dict = {block_id: off for block_id, off in zip(block_list, offsets)}
    save_path = os.path.join(tmp_folder,
                             'connected_components_offsets_%i.json' % job_id)
    with open(save_path, 'w') as f:
        json.dump(offset_dict, f)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    block_components(job_id, path)
