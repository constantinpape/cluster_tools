#! /bin/python

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


#
# Evaluation Tasks
#

class BlockTablesBase(luigi.Task):
    """ BlockTables base class
    """

    task_name = 'block_tables'
    src_file = os.path.abspath(__file__)
    allow_retries = False

    seg_path = luigi.Parameter()
    seg_key = luigi.Parameter()
    gt_path = luigi.Parameter()
    gt_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'seg_path': self.seg_path,
                       'seg_key': self.seg_key,
                       'gt_path': self.gt_path,
                       'gt_key': self.gt_key,
                       'block_shape': block_shape,
                       'output_path': self.output_path,
                       'output_key': self.output_key})

        shape = vu.get_shape(self.seg_path, self.seg_key)
        gt_shape = vu.get_shape(self.gt_path, self.gt_key)
        assert shape == gt_shape, "%s, %s" % (str(shape), str(gt_shape))

        # create output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape,
                              dtype='uint64',
                              chunks=tuple(block_shape),
                              compression='gzip')

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class BlockTablesLocal(BlockTablesBase, LocalTask):
    """ BlockTables on local machine
    """
    pass


class BlockTablesSlurm(BlockTablesBase, SlurmTask):
    """ BlockTables on slurm cluster
    """
    pass


class BlockTablesLSF(BlockTablesBase, LSFTask):
    """ BlockTables on lsf cluster
    """
    pass


#
# Implementation
#

def _block_table(block_id, blocking, ds_seg, ds_gt,
                 out_path, ignore_seg, ignore_gt):
    fu.log("start processing block %i" % block_id)
    # read labels and input in this block
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    seg = ds_seg[bb]
    gt = ds_gt[bb]

    # check if segmentation block is empty
    if (gt != ignore_gt).sum() == 0:
        fu.log("block %i is empty" % block_id)
        fu.log_block_success(block_id)
        return

    chunk_id = tuple(beg // ch
                     for beg, ch in zip(block.begin,
                                        blocking.blockShape))
    ndist.computeAndSerializeContingencyTable(seg, gt,
                                              out_path, chunk_id,
                                              ignore_seg, ignore_gt)
    fu.log_block_success(block_id)


def block_tables(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    seg_path = config['seg_path']
    seg_key = config['seg_key']
    gt_path = config['gt_path']
    gt_key = config['gt_key']

    output_path = config['output_path']
    output_key = config['output_key']
    out_path = os.path.join(output_path, output_key)
    fu.log("Seriailze results to %s" % out_path)

    block_shape = config['block_shape']
    block_list = config['block_list']

    ignore_seg = config.get("ignore_seg", 0)
    ignore_gt = config.get("ignore_gt", 0)

    with vu.file_reader(seg_path, 'r') as f:
        shape = f[seg_key].shape

    blocking = nt.blocking([0, 0, 0],
                           list(shape),
                           list(block_shape))

    with vu.file_reader(seg_path, 'r') as f_seg,\
            vu.file_reader(gt_path, 'r') as f_gt:
        ds_seg = f_seg[seg_key]
        ds_gt = f_gt[gt_key]

        [_block_table(block_id, blocking,
                      ds_seg, ds_gt, out_path,
                      ignore_seg, ignore_gt)
         for block_id in block_list]

    # write out the number of labels for convinience
    if job_id == 0:
        n_labels_seg = vu.file_reader(seg_path, 'r')[seg_key].attrs['maxId'] + 1
        n_labels_gt = vu.file_reader(gt_path, 'r')[gt_key].attrs['maxId'] + 1
        with vu.file_reader(output_path) as f:
            attrs = f[output_key].attrs
            attrs['n_labels_seg'] = int(n_labels_seg)
            attrs['n_labels_gt'] = int(n_labels_gt)
            attrs['n_points'] = int(np.prod(list(shape)))

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    block_tables(job_id, path)
