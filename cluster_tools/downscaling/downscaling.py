#! /bin/python

import os
import sys
import json

import numpy as np
import luigi
import vigra
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# downscaling tasks
#

# TODO which algorithms do we support ?
# check scipy
# detail-preserving downscaling ?
class DownscalingBase(luigi.Task):
    """ downscaling base class
    """

    task_name = 'downscaling'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # the scale used to downsample the data.
    # can be list to account for anisotropic downsacling
    # however this is not supported by all implementations
    scale_factor = luigi.ListParameter()
    # scale prefix for unique task identifier
    scale_prefix = luigi.Parameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'library': 'vigra', 'chunks': None, 'compression': 'gzip',
                       'halo': None, 'library_kwargs': None})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def downsample_shape(self, shape):
        new_shape = tuple(sh // sf if sh % sf == 0 else sh // sf + (sf - sh % sf)
                          for sh, sf in zip(shape, self.scale_factor))

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            prev_shape = f[self.input_key].shape
            dtype = f[self.input_key].dtype
        assert len(prev_shape) == 3, "Only support 3d inputs"
        assert dtype in ('float32', 'float64', 'uint8', 'uint16'), "Need float, byte or short input, got %s" % dtype
        shape = self.downsample_shape(prev_shape)

        # load the downscaling config
        task_config = self.get_task_config()

        # get the scale factor and check if we
        # do isotropic scaling
        scale_factor = self.scale_factor
        if all(sf == scale_factor[0] for sf in scale_factor):
            scale_factor = scale_factor[0]
        else:
            # for now, we only support downscaling in-plane inf the scale-factor
            # is anisotropic
            assert scale_factor[0] == 1
            assert scale_factor[1] == scale_factor[2]

        # read the output chunks
        chunks = task_config.pop('chunks', None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)
        else:
            # TODO verify chunks further
            assert len(chunks) == 3, "Chunks must be 3d"

        compression = task_config.pop('compression', 'gzip')
        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression=compression, dtype=dtype)

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key,
                            'block_shape': block_shape, 'scale_factor': scale_factor})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config, self.scale_prefix)
        self.submit_jobs(n_jobs, self.scale_prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.scale_prefix)
        self.check_jobs(n_jobs, self.scale_prefix)


class DownscalingLocal(DownscalingBase, LocalTask):
    """
    downscaling on local machine
    """
    pass


class DownscalingSlurm(DownscalingBase, SlurmTask):
    """
    downscaling on slurm cluster
    """
    pass


class DownscalingLSF(DownscalingBase, LSFTask):
    """
    downscaling on lsf cluster
    """
    pass


#
# Implementation
#


def _ds_block(blocking, block_id, ds_in, ds_out, scale_factor, halo, library_kwargs):
    fu.log("start processing block %i" % block_id)

    # load the block (output dataset / downsampled) coordinates
    if halo is None:
        block = blocking.getBlock(block_id)
        local_bb = np.s_[:]
        in_bb = vu.block_to_bb(block)
        out_bb = vu.block_to_bb(block)
        out_shape = block.shape
    else:
        halo_ds = [ha // scale_factor] if isinstance(scale_factor, int) else\
            [ha // sf for sf in scale_factor]
        block = blocking.getBlockWithHalo(block_id, halo_ds)
        in_bb = vu.block_to_bb(block.outerBlock)
        out_bb = vu.block_to_bb(block.innerBlock)
        local_bb = vu.block_to_bb(block.innerBlockLocal)
        out_shape = block.outerBlock.shape

    # upsample the input bounding box
    if isinstance(sf, int):
        in_bb = tuple(slice(ib.start * scale_factor, min(ib.stop * scale_factor, sh))
                      for ib, sh in zip(in_bb, ds_in.shape))
    else:
        in_bb = tuple(slice(ib.start * sf, min(ib.stop * sf, sh))
                      for ib, sf, sh in zip(in_bb, scale_factor, ds_in.shape))

    x = ds_in[in_bb]
    dtype = x.dtype
    if np.dtype(dtype) != np.dtype('float32'):
        x = x.astype('float32')

    if isinstance(sf, int):
        out = vigra.sampling.resize(x, shape=out_shape, **library_kwargs)
    else:
        out = np.zeros(out_shape, dtype='float32')
        for z in range(out_shape[0]):
            out[z] = vigra.sampling.resize(x[z], shape=out_shape[1:], **library_kwargs)

    if np.dtype(dtype) in (np.dtype('uint8'), np.dtype('uint16')):
        out = np.round(out)

    ds_out[out_bb] = out[local_bb].astype(dtype)

    # log block success
    fu.log_block_success(block_id)


def downscaling(job_id, config_path):
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
    # TODO support more impls
    assert config.get('library', 'vigra') == 'vigra'
    halo = config.get('halo', None)
    library_kwargs = config.get(library_kwargs, None)
    if library_kwargs is None:
        library_kwargs = {}

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in  = f_in[input_key]
        ds_out = f_out[output_key]

        shape = ds_out.shape
        # get the blocking
        blocking = nt.blocking([0, 0, 0], shape, block_shape)

        for block_id in block_list:
            _ds_block(blocking, block_id, ds_in, ds_out,
                      scale_factor, halo, library_kwargs)
    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    downscaling(job_id, path)
