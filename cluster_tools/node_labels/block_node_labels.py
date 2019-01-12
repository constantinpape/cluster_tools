#! /bin/python

import os
import sys
import json
import numpy as np

import luigi
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class BlockNodeLabelsBase(luigi.Task):
    """ BlockNodeLabels base class
    """

    task_name = 'block_node_labels'
    src_file = os.path.abspath(__file__)

    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
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
        config.update({'ws_path': self.ws_path, 'ws_key': self.ws_key,
                       'input_path': self.input_path,
                       'input_key': self.input_key,
                       'block_shape': block_shape,
                       'output_path': self.output_path, 'output_key': self.output_key})

        shape = vu.get_shape(self.ws_path, self.ws_key)
        chunks = tuple(min(bs, sh) for bs, sh in zip(block_shape, shape))

        # create output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape,
                              dtype='uint64',
                              chunks=chunks,
                              compression='gzip')

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape,
                                             roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class BlockNodeLabelsLocal(BlockNodeLabelsBase, LocalTask):
    """ BlockNodeLabels on local machine
    """
    pass


class BlockNodeLabelsSlurm(BlockNodeLabelsBase, SlurmTask):
    """ BlockNodeLabels on slurm cluster
    """
    pass


class BlockNodeLabelsLSF(BlockNodeLabelsBase, LSFTask):
    """ BlockNodeLabels on lsf cluster
    """
    pass


#
# Implementation
#

def _labels_for_block(block_id, blocking,
                      ds_ws, out_path, labels):
    fu.log("start processing block %i" % block_id)
    # read labels and input in this block
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    ws = ds_ws[bb]

    # check if watershed block is empty
    if ws.sum() == 0:
        fu.log("watershed of block %i is empty" % block_id)
        fu.log_block_success(block_id)
        return

    # serialize the overlaps
    labs = labels[bb].astype('uint64')

    # check if label block is empty:
    if labs.sum() == 0:
        fu.log("labels of block %i is empty" % block_id)
        fu.log_block_success(block_id)
        return

    chunk_id = tuple(beg // ch
                     for beg, ch in zip(block.begin,
                                        blocking.blockShape))
    ndist.computeAndSerializeLabelOverlaps(ws, labs,
                                           out_path, chunk_id)
    fu.log_block_success(block_id)


def block_node_labels(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    ws_path = config['ws_path']
    ws_key = config['ws_key']
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    block_shape = config['block_shape']
    block_list = config['block_list']

    with vu.file_reader(ws_path, 'r') as f:
        shape = f[ws_key].shape

    blocking = nt.blocking([0, 0, 0],
                           list(shape),
                           list(block_shape))

    # labels can either be interpolated or full volume
    f_lab = vu.file_reader(input_path, 'r')
    ds_labels = f_lab[input_key]
    lab_shape = ds_labels.shape
    # label shape is smaller than ws shape
    # -> interpolated
    if all(lsh < sh for lsh, sh in zip(lab_shape, shape)):
        labels = vu.InterpolatedVolume(ds_labels, shape, spline_order=0)
        f_lab.close()
    else:
        assert lab_shape == shape
        labels = ds_labels

    with vu.file_reader(ws_path, 'r') as f_in:
        ds_ws = f_in[ws_key]
        out_path = os.path.join(output_path, output_key)
        [_labels_for_block(block_id, blocking,
                           ds_ws, out_path, labels)
         for block_id in block_list]
        max_id = ds_ws.attrs['maxId']

    # need to serialize the label max-id here for
    # the merge_node_labels task
    if job_id == 0:
        with vu.file_reader(output_path) as f:
            ds_out = f[output_key]
            ds_out.attrs['maxId'] = max_id

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    block_node_labels(job_id, path)
