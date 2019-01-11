#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# WatershedFromSeeds Tasks
#

class WatershedFromSeedsBase(luigi.Task):
    """ WatershedFromSeeds base class
    """

    task_name = 'watershed_from_seeds'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    seeds_path = luigi.Parameter()
    seeds_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'channel_begin': 0, 'channel_end': None,
                       'agglomerate_channels': 'mean', 'size_filter': 0})
        return config

    def clean_up_for_retry(self, block_list):
        # TODO remove any output of failed blocks because it might be corrupted
        super().clean_up_for_retry(block_list)

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)
        if len(shape) == 4:
            shape = shape[1:]

        # load the watershed config
        ws_config = self.get_task_config()

        # require output dataset
        # TODO read chunks from config
        chunks = tuple(bs // 2 for bs in block_shape)
        chunks = tuple(min(ch, sh) for ch, sh in zip(chunks, shape))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # update the config with input and output paths and keys
        # as well as block shape
        ws_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                          'seeds_path': self.seeds_path, 'seeds_key': self.seeds_key,
                          'output_path': self.output_path, 'output_key': self.output_key,
                          'block_shape': block_shape})
        if self.mask_path != '':
            assert self.mask_key != ''
            ws_config.update({'mask_path': self.mask_path, 'mask_key': self.mask_key})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        self._write_log('scheduling %i blocks to be processed' % len(block_list))

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, ws_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class WatershedFromSeedsLocal(WatershedFromSeedsBase, LocalTask):
    """
    WatershedFromSeeds on local machine
    """
    pass


class WatershedFromSeedsSlurm(WatershedFromSeedsBase, SlurmTask):
    """
    WatershedFromSeeds on slurm cluster
    """
    pass


class WatershedFromSeedsLSF(WatershedFromSeedsBase, LSFTask):
    """
    WatershedFromSeeds on lsf cluster
    """
    pass


#
# Implementation
#

def _read_data(ds_in, input_bb, config):
    # read the input data
    if ds_in.ndim == 4:
        channel_begin = config.get('channel_begin', 0)
        channel_end = config.get('channel_end', None)
        input_bb = (slice(channel_begin, channel_end),) + input_bb
        input_ = vu.normalize(ds_in[input_bb])
        agglomerate = config.get('agglomerate_channels', 'mean')
        assert agglomerate in ('mean', 'max', 'min')
        input_ = getattr(np, agglomerate)(input_, axis=0)
    else:
        input_ = vu.normalize(ds_in[input_bb])
    return input_


def _ws_block(blocking, block_id, ds_in, ds_seeds, ds_out, config):
    fu.log("start processing block %i" % block_id)
    size_filter = config.get('size_filter', 0)

    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    # read input and seeds
    input_ = _read_data(ds_in, bb, config)
    seeds = ds_seeds[bb]

    # run watershed
    # vigra seeds need to be uint32
    # TODO use skimage or make PR to vigra
    max_id = int(seeds.max())
    # TODO implment mapping of ids ...
    assert max_id < np.iinfo('uint32').max, "Overflow detected"
    ws, _ = vu.watershed(input_, seeds=seeds.astype('uint32'),
                         size_filter=size_filter)
    ds_out[bb] = ws.astype('uint64')

    # log block success
    fu.log_block_success(block_id)


def _ws_block_masked(blocking, block_id,
                     ds_in, ds_seeds, ds_out, mask, config):
    fu.log("start processing block %i" % block_id)
    size_filter = config.get('size_filter', 0)

    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    # get the mask and check if we have any pixels
    in_mask = mask[bb].astype('bool')
    if np.sum(in_mask) == 0:
        fu.log_block_success(block_id)
        return

    # read input and seeds
    input_ = _read_data(ds_in, bb, config)
    seeds = ds_seeds[bb]

    # mask the input and run watershed
    inv_mask = np.logical_not(in_mask)
    input_[inv_mask] = 1

    # vigra seeds need to be uint32
    # TODO use skimage or make PR to vigra
    max_id = int(seeds.max())
    # TODO implment mapping of ids ...
    assert max_id < np.iinfo('uint32').max, "Overflow detected"
    ws, _ = vu.watershed(input_, seeds=seeds.astype('uint32'),
                         size_filter=size_filter)
    ws = ws.astype('uint64')

    # set mask to zero and write ws
    ws[inv_mask] = 0
    ds_out[bb] = ws

    # log block success
    fu.log_block_success(block_id)


def watershed_from_seeds(job_id, config_path):
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

    # TODO seeds and output might be identical
    # in that case we would need in-place logic if we
    # want to support h5 (it's fine with n5 as is)
    # read the seed and  output config
    seeds_path = config['seeds_path']
    seeds_key = config['seeds_key']
    output_path = config['output_path']
    output_key = config['output_key']

    # check if we have a mask
    with_mask = 'mask_path' in config
    if with_mask:
        mask_path = config['mask_path']
        mask_key = config['mask_key']

    # get the blocking
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in,\
         vu.file_reader(seeds_path, 'r') as f_seeds,\
         vu.file_reader(output_path) as f_out:

        ds_in  = f_in[input_key]
        assert ds_in.ndim in (3, 4)
        ds_seeds = f_out[seeds_key]
        assert ds_seeds.ndim == 3
        ds_out = f_out[output_key]
        assert ds_out.ndim == 3

        # note that the mask is usually small enough to keep it
        # in memory (and we interpolate to get to the full volume)
        # if this does not hold need to change this code!
        if with_mask:
            mask = vu.load_mask(mask_path, mask_key, shape)
            for block_id in block_list:
                _ws_block_masked(blocking, block_id,
                                 ds_in, ds_seeds, ds_out, mask, config)

        else:
            for block_id in block_list:
                _ws_block(blocking, block_id, ds_in, ds_seeds, ds_out, config)
    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    watershed_from_seeds(job_id, path)
