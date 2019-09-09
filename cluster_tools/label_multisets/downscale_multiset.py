#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt
from elf.util import chunks_overlapping_roi, downscale_shape

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
from cluster_tools.utils.numpy_utils import set_numpy_threads
set_numpy_threads(1)
import numpy as np
from elf.label_multiset import (deserialize_multiset, serialize_multiset,
                                downsample_multiset, merge_multisets,
                                LabelMultiset)


class DownscaleMultisetBase(luigi.Task):
    """ DownscaleMultiset base class
    """

    task_name = 'downscale_multiset'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # scale factors and restrict set
    scale_factor = luigi.Parameter()
    effective_scale_factor = luigi.Parameter()
    scale_prefix = luigi.Parameter()
    restrict_set = luigi.Parameter()
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

        # load the downscale_multiset config
        config = self.get_task_config()

        compression = config.get('compression', 'gzip')
        out_shape = downscale_shape(shape, self.scale_factor)
        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=out_shape, chunks=tuple(block_shape),
                              compression=compression, dtype='uint8')

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'scale_factor': self.scale_factor, 'restrict_set': self.restrict_set,
                       'effective_scale_factor': self.effective_scale_factor, 'block_shape': block_shape})
        block_list = vu.blocks_in_volume(out_shape, block_shape, roi_begin, roi_end)
        self._write_log('scheduling %i blocks to be processed' % len(block_list))
        n_jobs = min(len(block_list), self.max_jobs)
        self._write_log("submitting %i blocks with %i jobs" % (len(block_list), n_jobs))

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config, self.scale_prefix)
        self.submit_jobs(n_jobs, self.scale_prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs, self.scale_prefix)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              '%s_%s.log' % (self.task_name, self.scale_prefix)))


class DownscaleMultisetLocal(DownscaleMultisetBase, LocalTask):
    """
    DownscaleMultiset on local machine
    """
    pass


class DownscaleMultisetSlurm(DownscaleMultisetBase, SlurmTask):
    """
    DownscaleMultiset on slurm cluster
    """
    pass


class DownscaleMultisetLSF(DownscaleMultisetBase, LSFTask):
    """
    DownscaleMultiset on lsf cluster
    """
    pass


#
# Implementation
#


def background_multiset(shape, effective_pixel_size):
    size = np.prod(list(shape))
    amax = np.zeros(size, dtype='uint64')
    offsets = np.zeros(size, dtype='int32')
    ids = np.zeros(1, dtype='uint64')
    counts = np.array([effective_pixel_size], dtype='int32')
    return LabelMultiset(amax, offsets, ids, counts, shape)


def normalize_chunks(chunk_ids):
    # find the leftmost chunk
    chunks = np.array(chunk_ids)
    origin_id = np.argmin(chunks.sum(axis=1))
    origin = chunks[origin_id]
    assert all((origin[i] <= chunks[:, i]).all() for i in range(chunks.shape[1]))
    chunk_ids = [tuple(ch - orig for ch, orig in zip(chunk_id, origin))
                 for chunk_id in chunk_ids]
    return chunk_ids


def _downscale_multiset_block(blocking, block_id, ds_in, ds_out,
                              blocking_prev, scale_factor, restrict_set,
                              effective_pixel_size):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)

    ndim = ds_in.ndim
    # get the blocks and chunk ids
    # corresponding to this block in the previous scale level
    roi_begin_prev = [beg * sc for beg, sc in zip(block.begin, scale_factor)]
    roi_end_prev = [min(end * sc, sh) for end, sc, sh in zip(block.end,
                                                             scale_factor,
                                                             ds_in.shape)]
    roi_shape_prev = tuple(re - rb for rb, re in zip(roi_begin_prev, roi_end_prev))
    roi_prev = tuple(slice(rb, re) for rb, re in zip(roi_begin_prev, roi_end_prev))

    block_shape_prev = blocking_prev.blockShape
    chunk_ids_prev = list(chunks_overlapping_roi(roi_prev, block_shape_prev))
    block_ids_prev = np.ravel_multi_index(np.array([[cid[d] for cid in chunk_ids_prev] for d in range(ndim)],
                                                   dtype='int'),
                                          blocking_prev.blocksPerAxis)
    blocks_prev = [blocking_prev.getBlock(bid) for bid in block_ids_prev]

    multisets = [ds_in.read_chunk(chunk_id) for chunk_id in chunk_ids_prev]
    # TODO can paintera deal with this ?
    if all(mset is None for mset in multisets):
        fu.log_block_success(block_id)
        return

    multisets = [background_multiset(block_prev.shape, effective_pixel_size) if mset is None
                 else deserialize_multiset(mset, block_prev.shape)
                 for mset, block_prev in zip(multisets, blocks_prev)]

    chunk_ids_prev = normalize_chunks(chunk_ids_prev)
    multiset = merge_multisets(multisets, chunk_ids_prev,
                               roi_shape_prev, blocking_prev.blockShape)

    # compute multiset from input labels
    multiset = downsample_multiset(multiset, scale_factor, restrict_set)
    ser = serialize_multiset(multiset)

    chunk_id = tuple(beg // ch for beg, ch in zip(block.begin, ds_out.chunks))
    ds_out.write_chunk(chunk_id, ser, True)
    fu.log_block_success(block_id)


def write_metadata(ds_out, max_id, restrict_set, scale_factor):
    attrs = ds_out.attrs
    attrs['maxId'] = max_id
    attrs['isLabelMultiset'] = True
    attrs['maxNumEntries'] = restrict_set
    # we reverse the scale factor, because java axis conventions are XYZ and we have ZYX
    attrs['downsamplingFactors'] = [float(sf) for sf in reversed(scale_factor)]


def downscale_multiset(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']
    restrict_set = config['restrict_set']
    restrict_set = -1 if restrict_set is None else restrict_set
    scale_factor = config['scale_factor']
    scale_factor = scale_factor if isinstance(scale_factor, list) else [scale_factor] * 3

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    # read the output config
    output_path = config['output_path']
    output_key = config['output_key']
    shape = list(vu.get_shape(output_path, output_key))
    prev_shape = list(vu.get_shape(input_path, input_key))

    # NOTE for now, we assume that the block_shape stays constant throughout
    # the scale levels
    # get the blocking for this scale level
    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    # get the blocking for the previous scale level
    blocking_prev = nt.blocking([0, 0, 0], prev_shape, block_shape)

    effective_scale_factor = config['effective_scale_factor']
    effective_pixel_size = int(np.prod(effective_scale_factor))

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        for block_id in block_list:
            _downscale_multiset_block(blocking, block_id, ds_in, ds_out,
                                      blocking_prev, scale_factor, restrict_set,
                                      effective_pixel_size)

        if job_id == 0:
            max_id = ds_in.attrs['maxId']
            write_metadata(ds_out, max_id, restrict_set, scale_factor)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    downscale_multiset(job_id, path)
