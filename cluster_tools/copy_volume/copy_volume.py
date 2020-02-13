#! /bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty.tools as nt
from concurrent import futures

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# copy tasks
#

class CopyVolumeBase(luigi.Task):
    """ copy_volume base class
    """

    task_name = 'copy_volume'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    prefix = luigi.Parameter()
    dtype = luigi.Parameter(default=None)
    fit_to_roi = luigi.BoolParameter(default=False)
    effective_scale_factor = luigi.ListParameter(default=[])
    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'chunks': None, 'compression': 'gzip',
                       'reduce_channels': None, 'map_uniform_blocks_to_background': False})
        return config

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape
            ds_dtype = f[self.input_key].dtype
            ds_chunks = f[self.input_key].chunks

        # load the config
        task_config = self.get_task_config()

        ndim = len(shape)
        assert ndim in (3, 4), "Copying is only supported for 3d and 4d inputs"
        # if we have a roi, we need to:
        # - scale the roi to the effective scale, if effective scale is given
        # - shrink the shape to the roi, if fit_to_roi is True
        if roi_begin is not None:
            assert ndim == 3, "Don't support roi for 4d yet"
            assert roi_end is not None
            if self.effective_scale_factor:
                roi_begin = [int(rb // sf)
                             for rb, sf in zip(roi_begin, self.effective_scale_factor)]
                roi_end = [int(re // sf)
                           for re, sf in zip(roi_end, self.effective_scale_factor)]

            if self.fit_to_roi:
                out_shape = tuple(roie - roib for roib, roie in zip(roi_begin, roi_end))
                # if we fit to roi, the task config needs to be updated with the roi begin,
                # because the output bounding boxes need to be offseted by roi_begin
                task_config.update({'roi_begin': roi_begin})
            else:
                out_shape = shape
        else:
            out_shape = shape

        if task_config.get('reduce_channels', None) is not None and len(out_shape) == 4:
            out_shape = out_shape[1:]

        compression = task_config.pop('compression', 'gzip')
        dtype = str(ds_dtype) if self.dtype is None else self.dtype

        chunks = task_config.pop('chunks', None)
        chunks = tuple(block_shape) if chunks is None else chunks
        if len(chunks) == 3 and ndim == 4:
            chunks = (ds_chunks[0],) + chunks
        assert all(bs % ch == 0 for bs, ch in zip(block_shape,
                                                  chunks[1:] if ndim == 4 else chunks))

        # require output dataset
        with vu.file_reader(self.output_path) as f:
            chunks = tuple(min(ch, sh) for ch, sh in zip(chunks, out_shape))
            f.require_dataset(self.output_key, shape=out_shape, chunks=chunks,
                              compression=compression, dtype=dtype)

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key,
                            'block_shape': block_shape, 'dtype': dtype})

        if len(shape) == 4:
            shape = shape[1:]

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        self._write_log("scheduled %i blocks to run" % len(block_list))

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config, self.prefix)
        self.submit_jobs(n_jobs, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.prefix)
        self.check_jobs(n_jobs, self.prefix)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.prefix))


class CopyVolumeLocal(CopyVolumeBase, LocalTask):
    """
    copy_volume local machine
    """
    pass


class CopyVolumeSlurm(CopyVolumeBase, SlurmTask):
    """
    copy on slurm cluster
    """
    pass


class CopyVolumeLSF(CopyVolumeBase, LSFTask):
    """
    copy_volume on lsf cluster
    """
    pass


#
# Implementation
#


def cast_type(data, dtype):
    if np.dtype(data.dtype) == np.dtype(dtype):
        return data
    # special casting for uint8
    elif np.dtype(dtype) == 'uint8':
        data = vu.normalize(data)
        data *= 255
        return data.astype('uint8')
    else:
        return data.astype(dtype)


def _copy_blocks(ds_in, ds_out, blocking, block_list, roi_begin, reduce_function, n_threads,
                 map_uniform_blocks_to_background):
    dtype = ds_out.dtype

    def _copy_block(block_id):
        fu.log("start processing block %i" % block_id)

        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        if ds_in.ndim == 4:
            bb = (slice(None),) + bb

        data = ds_in[bb]
        # don't write empty blocks
        if data.sum() == 0:
            fu.log_block_success(block_id)
            return

        if map_uniform_blocks_to_background and (len(np.unique(data)) == 1):
            fu.log_block_success(block_id)
            return

        # if we have a roi begin, we need to substract it
        # from the output bounding box, because in this case
        # the output shape has been fit to the roi
        if roi_begin is not None:
            bb = tuple(slice(b.start - off, b.stop - off)
                       for b, off in zip(bb, roi_begin))

        if reduce_function is not None and data.ndim == 4:
            data = reduce_function(data[0:3], axis=0)
            bb = bb[1:]

        ds_out[bb] = cast_type(data, dtype)
        fu.log_block_success(block_id)

    if n_threads > 1:
        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_copy_block, block_id) for block_id in block_list]
            [t.result() for t in tasks]
    else:
        [_copy_block(block_id) for block_id in block_list]


def copy_volume(job_id, config_path):
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

    # check if we offset by roi
    roi_begin = config.get('roi_begin', None)

    # check if we reduce channels
    reduce_function = config.get('reduce_channels', None)
    if reduce_function is not None:
        reduce_function = getattr(np, reduce_function)

    map_uniform_blocks_to_background = config.get('map_uniform_blocks_to_background', False)
    n_threads = config.get('threads_per_job', 1)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        shape = list(ds_in.shape)
        if len(shape) == 4:
            shape = shape[1:]
        blocking = nt.blocking([0, 0, 0], shape, block_shape)
        _copy_blocks(ds_in, ds_out, blocking, block_list, roi_begin,
                     reduce_function, n_threads, map_uniform_blocks_to_background)

        # copy the attributes with job 0
        if job_id == 0:
            attrs_in = ds_in.attrs
            attrs_out = ds_out.attrs
            for k, v in attrs_in.items():
                attrs_out[k] = v

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    copy_volume(job_id, path)
