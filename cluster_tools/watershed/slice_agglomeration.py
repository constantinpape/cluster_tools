#! /bin/python

import os
import sys
import json
from concurrent import futures

import luigi
import numpy as np
import nifty.tools as nt

import elf.segmentation.features as feats
import elf.segmentation.multicut as mc
from vigra.analysis import relabelConsecutive

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# SliceAgglomeration Tasks
#

class SliceAgglomerationBase(luigi.Task):
    """ SliceAgglomeration base class
    """

    task_name = 'slice_agglomeration'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    have_ignore_label = luigi.BoolParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        config.update({'beta_stitch': 0.5,
                       'beta': 0.75,
                       'invert_inputs': False,
                       'channel_begin': None,
                       'channel_end': None})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get volume shape and chunks
        with vu.file_reader(self.output_path, 'r') as f:
            shape = f[self.output_key].shape
            chunks = f[self.output_key].chunks
        assert len(shape) == 3

        # load the slice_agglomeration config
        config = self.get_task_config()

        # we deal with different block shapes:
        # - block_shape: the block shape used for watershed calculation
        # - slice_shape: the (2d) shape of a single slice
        # - slice_block_shape: the watershed volume chunks in z + full slice shape
        slice_shape = shape[1:]
        slice_block_shape = (chunks[0],) + slice_shape

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': slice_block_shape, 'have_ignore_label': self.have_ignore_label,
                       'block_shape_2d': block_shape[1:]})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, slice_block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        self._write_log('scheduling %i blocks to be processed' % len(block_list))
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class SliceAgglomerationLocal(SliceAgglomerationBase, LocalTask):
    """
    SliceAgglomeration on local machine
    """
    pass


class SliceAgglomerationSlurm(SliceAgglomerationBase, SlurmTask):
    """
    SliceAgglomeration on slurm cluster
    """
    pass


class SliceAgglomerationLSF(SliceAgglomerationBase, LSFTask):
    """
    SliceAgglomeration on lsf cluster
    """
    pass


#
# Implementation
#


def agglomerate_slice(seg, input_, z, config):
    # check if this slic is empty
    if np.sum(seg) == 0:
        return seg

    have_ignore_label = config['have_ignore_label']
    block_shape = config['block_shape_2d']

    foreground_mask = seg != 0

    # relabel the segmentation
    _, max_id, _ = relabelConsecutive(seg, out=seg, keep_zeros=True, start_label=1)
    seg = seg.astype('uint32')

    # construct rag and get edge features
    rag = feats.compute_rag(seg, n_labels=max_id + 1, n_threads=1)
    edge_features = feats.compute_boundary_mean_and_length(rag, input_, n_threads=1)[:, 0]

    # set edges to ignore label to be maximally repulsive
    if have_ignore_label:
        uv_ids = rag.uvIds()
        ignore_mask = (uv_ids == 0).any(axis=1)
        edge_features[ignore_mask] = 1

    # get the stitiching edges
    stitch_edges = feats.get_stitch_edges(rag, seg, block_shape)
    beta1 = config.get('beta_stitch', 0.5)
    beta2 = config.get('beta', 0.75)

    costs = np.zeros_like(edge_features)
    costs[stitch_edges] = mc.compute_edge_costs(edge_features[stitch_edges], beta=beta1)
    costs[~stitch_edges] = mc.compute_edge_costs(edge_features[~stitch_edges], beta=beta2)
    node_labels = mc.multicut_kernighan_lin(rag, costs)

    node_labels, max_id, _ = relabelConsecutive(node_labels, start_label=1, keep_zeros=True)

    # project node labels back to segmentation
    seg = feats.project_node_labels_to_pixels(rag, node_labels, n_threads=1).astype('uint64')

    # we change to slice base id offset
    id_offset = z * np.prod(list(seg.shape))
    assert id_offset < np.iinfo('uint64').max, "Id overflow"

    # add offset back to segmentation
    seg[foreground_mask] += id_offset
    return seg


def _slice_agglomeration(blocking, block_id, ds_in, ds_out, config):
    fu.log("start processing block %i" % block_id)

    n_threads = config['threads_per_job']
    ds_in.n_threads = n_threads
    ds_out.n_threads = n_threads

    invert_inputs = config.get('invert_inputs', False)

    bb = vu.block_to_bb(blocking.getBlock(block_id))
    # load the segmentation / output
    seg = ds_out[bb]

    # load the input data
    ndim_in = ds_in.ndim
    # we are conservative and always accumulate affinities with max
    if ndim_in == 4:
        channel_begin = config.get('channel_begin', None)
        channel_end = config.get('channel_end', None)
        channel_begin = 0 if channel_begin is None else channel_begin
        channel_end = ds_in.shape[0] if channel_end is None else channel_end
        bb_in = (slice(channel_begin, channel_end),) + bb
        input_ = ds_in[bb_in]
        if invert_inputs:
            input_ = 1. - input_
        input_ = vu.normalize(np.max(input_, axis=0))
    else:
        input_ = vu.normalize(ds_in[bb])
        if invert_inputs:
            input_ = 1. - input_

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(agglomerate_slice, seg[z], input_[z], z, config)
                 for z in range(seg.shape[0])]
        slice_segs = [t.result() for t in tasks]

    seg = np.concatenate([sseg[None] for sseg in slice_segs], axis=0)
    ds_out[bb] = seg

    # log block success
    fu.log_block_success(block_id)


def slice_agglomeration(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']
    shape = list(vu.get_shape(input_path, input_key))
    if len(shape) == 4:
        shape = shape[1:]

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    # read the output config
    output_path = config['output_path']
    output_key = config['output_key']

    # get the blocking
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        assert ds_in.ndim in (3, 4)
        ds_out = f_out[output_key]
        assert ds_out.ndim == 3
        for block_id in block_list:
            _slice_agglomeration(blocking, block_id, ds_in, ds_out, config)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    slice_agglomeration(job_id, path)
