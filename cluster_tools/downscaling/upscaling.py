#! /bin/python

import os
import sys
import json
from functools import partial
from concurrent import futures
from math import ceil

# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
from elf.util import set_numpy_threads
set_numpy_threads(1)
import numpy as np

import luigi
import vigra
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# downscaling tasks
#


class UpscalingBase(luigi.Task):
    """ upscaling base class
    """

    task_name = 'upscaling'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # the scale used to downsample the data.
    # can be list to account for anisotropic downsacling
    scale_factor = luigi.Parameter()
    # scale prefix for unique task identifier
    scale_prefix = luigi.Parameter(default='')
    effective_scale_factor = luigi.ListParameter(default=[])
    dependency = luigi.TaskParameter(default=DummyTask())

    interpolatable_types = ('float32', 'float64', 'uint8', 'uint16')

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'library': 'vigra', 'chunks': None, 'compression': 'gzip',
                       'library_kwargs': None})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def upsample_shape(self, shape, scale_factor):
        if isinstance(scale_factor, (list, tuple)):
            new_shape = tuple(int(sh * sf) for sh, sf in zip(shape, scale_factor))
        else:
            new_shape = tuple(int(sh * scale_factor) for sh in shape)
        return new_shape

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            prev_shape = f[self.input_key].shape
            dtype = f[self.input_key].dtype
        assert len(prev_shape) == 3, "Only support 3d inputs"

        # get the scale factor and check if we
        # do isotropic scaling
        scale_factor = self.scale_factor
        if isinstance(scale_factor, int):
            pass
        elif all(sf == scale_factor[0] for sf in scale_factor):
            assert len(scale_factor) == 3
            scale_factor = scale_factor[0]
        else:
            assert len(scale_factor) == 3
            # for now, we only support downscaling in-plane inf the scale-factor
            # is anisotropic
            assert scale_factor[0] == 1
            assert scale_factor[1] == scale_factor[2]

        shape = self.upsample_shape(prev_shape, scale_factor)
        self._write_log('upsampling with factor %s from shape %s to %s' % (str(scale_factor),
                                                                           str(prev_shape), str(shape)))

        # load the downscaling config
        task_config = self.get_task_config()

        # make sure that we have order 0 downscaling if our datatype is not interpolatable
        library = task_config.get('library', 'vigra')
        assert library == 'vigra'
        if dtype not in self.interpolatable_types:
            opts = task_config.get('library_kwargs', {})
            opts = {} if opts is None else opts
            order = opts.get('order', None)
            assert order == 0,\
                "datatype %s is not interpolatable, set 'library_kwargs' = {'order': 0} to downscale it" % dtype

        # read the output chunks
        chunks = task_config.pop('chunks', None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)
        else:
            chunks = tuple(chunks)
            # TODO verify chunks further
            assert len(chunks) == 3, "Chunks must be 3d"
        chunks = tuple(min(ch, sh) for sh, ch in zip(shape, chunks))

        compression = task_config.pop('compression', 'gzip')
        # require output dataset
        self._write_log("requiring output dataset @ %s:%s" % (self.output_path, self.output_key))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression=compression, dtype=dtype)

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key,
                            'block_shape': block_shape, 'scale_factor': scale_factor})

        # if we have a roi, we need to re-sample it
        if roi_begin is not None:
            assert roi_end is not None
            effective_scale = self.effective_scale_factor if self.effective_scale_factor else scale_factor
            self._write_log("upscaling roi with effective scale %s" % str(effective_scale))
            self._write_log("ROI before scaling: %s to %s" % (str(roi_begin), str(roi_end)))
            if isinstance(effective_scale, int):
                roi_begin = [rb * effective_scale for rb in roi_begin]
                roi_end = [re * effective_scale if re is not None else sh
                           for re, sh in zip(roi_end, shape)]
            else:
                roi_begin = [rb * sf for rb, sf in zip(roi_begin, effective_scale)]
                roi_end = [re * sf if re is not None else sh
                           for re, sf, sh in zip(roi_end, effective_scale, shape)]
            roi_begin = map(int, roi_begin)
            roi_end = map(int, roi_end)
            self._write_log("ROI after scaling: %s to %s" % (str(roi_begin), str(roi_end)))

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        self._write_log("scheduled %i blocks to run" % len(block_list))

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config, self.scale_prefix)
        self.submit_jobs(n_jobs, self.scale_prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.scale_prefix)
        self.check_jobs(n_jobs, self.scale_prefix)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.scale_prefix))


class UpscalingLocal(UpscalingBase, LocalTask):
    """
    downscaling on local machine
    """
    pass


class UpscalingSlurm(UpscalingBase, SlurmTask):
    """
    downscaling on slurm cluster
    """
    pass


class UpscalingLSF(UpscalingBase, LSFTask):
    """
    downscaling on lsf cluster
    """
    pass


#
# Implementation
#


def _upsample_block(blocking, block_id, ds_in, ds_out, scale_factor, sampler):
    fu.log("start processing block %i" % block_id)

    # load the block (output dataset / upscaled) coordinates
    block = blocking.getBlock(block_id)
    local_bb = np.s_[:]
    in_bb = vu.block_to_bb(block)
    out_bb = vu.block_to_bb(block)
    out_shape = block.shape

    # upsample the input bounding box
    if isinstance(scale_factor, int):
        in_bb = tuple(slice(int(ib.start // scale_factor),
                            min(int(ceil(ib.stop / scale_factor)), sh))
                      for ib, sh in zip(in_bb, ds_in.shape))
    else:
        in_bb = tuple(slice(int(ib.start // sf),
                            min(int(ceil(ib.stop // sf)), sh))
                      for ib, sf, sh in zip(in_bb, scale_factor, ds_in.shape))

    x = ds_in[in_bb]

    # don't sample empty blocks
    if np.sum(x != 0) == 0:
        fu.log_block_success(block_id)
        return

    dtype = x.dtype
    if np.dtype(dtype) != np.dtype('float32'):
        x = x.astype('float32')

    if isinstance(scale_factor, int):
        out = sampler(x, shape=out_shape)
    else:
        out = np.zeros(out_shape, dtype='float32')
        for z in range(out_shape[0]):
            out[z] = sampler(x[z], shape=out_shape[1:])

    if np.dtype(dtype) in (np.dtype('uint8'), np.dtype('uint16')):
        max_val = np.iinfo(np.dtype(dtype)).max
        np.clip(out, 0, max_val, out=out)
        np.round(out, out=out)

    try:
        ds_out[out_bb] = out[local_bb].astype(dtype)
    except IndexError:
        raise(IndexError("%s, %s, %s" % (str(out_bb), str(local_bb), str(out.shape))))

    # log block success
    fu.log_block_success(block_id)


def _submit_blocks(ds_in, ds_out, block_shape, block_list,
                   scale_factor, library,
                   library_kwargs, n_threads):

    # get the blocking
    shape = ds_out.shape
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    sampler = partial(vigra.sampling.resize, **library_kwargs)

    if n_threads <= 1:
        for block_id in block_list:
            _upsample_block(blocking, block_id, ds_in, ds_out,
                            scale_factor, sampler)
    else:
        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_upsample_block, blocking, block_id, ds_in, ds_out,
                               scale_factor, sampler) for block_id in block_list]
            [t.result() for t in tasks]


def upscaling(job_id, config_path):
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

    scale_factor = config['scale_factor']
    library = config.get('library', 'vigra')
    library_kwargs = config.get('library_kwargs', None)
    if library_kwargs is None:
        library_kwargs = {}
    n_threads = config.get('threads_per_job', 1)

    # submit blocks
    # check if in and out - file are the same
    # because hdf5 does not like opening files twice
    if input_path == output_path:
        with vu.file_reader(output_path) as f:
            ds_in = f[input_key]
            ds_out = f[output_key]
            _submit_blocks(ds_in, ds_out, block_shape, block_list, scale_factor,
                           library, library_kwargs, n_threads)

    else:
        with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
            ds_in = f_in[input_key]
            ds_out = f_out[output_key]
            _submit_blocks(ds_in, ds_out, block_shape, block_list, scale_factor,
                           library, library_kwargs, n_threads)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    upscaling(job_id, path)
