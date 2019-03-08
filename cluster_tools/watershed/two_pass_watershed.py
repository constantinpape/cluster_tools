#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import vigra
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.watershed.watershed import _ws_block, _get_bbs, _read_data, _apply_dt, _make_seeds, _make_hmap


#
# TwoPassWatershed Tasks
#

class TwoPassWatershedBase(luigi.Task):
    """ TwoPassWatershed base class
    """

    task_name = 'two_pass_watershed'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'threshold': .5,
                       'apply_dt_2d': True, 'pixel_pitch': None,
                       'apply_ws_2d': True, 'sigma_seeds': 2., 'size_filter': 25,
                       'sigma_weights': 2., 'halo': [0, 0, 0],
                       'channel_begin': 0, 'channel_end': None,
                       'agglomerate_channels': 'mean', 'alpha': 0.8,
                       'invert_inputs': False, 'non_maximum_suppression': True})
        return config

    def _ws_pass(self, block_list, config, prefix):
        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config, prefix)
        self.submit_jobs(n_jobs, prefix)
        # wait till jobs finish and check for job success
        self.wait_for_jobs(prefix)
        self.check_jobs(n_jobs, prefix)

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end, block_list_path = self.global_config_values(True)
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
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # update the config with input and output paths and keys
        # as well as block shape
        ws_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                          'output_path': self.output_path, 'output_key': self.output_key,
                          'block_shape': block_shape})

        blocking = nt.blocking([0, 0, 0], list(shape), list(block_shape))
        block_lists = vu.make_checkerboard_block_lists(blocking, roi_begin, roi_end)
        for pass_id, block_list in enumerate(block_lists):
            ws_config['pass'] = pass_id
            self._ws_pass(block_list, ws_config, 'pass_%i' % pass_id)


class TwoPassWatershedLocal(TwoPassWatershedBase, LocalTask):
    """
    TwoPassWatershed on local machine
    """
    pass


class TwoPassWatershedSlurm(TwoPassWatershedBase, SlurmTask):
    """
    TwoPassWatershed on slurm cluster
    """
    pass


class TwoPassWatershedLSF(TwoPassWatershedBase, LSFTask):
    """
    TwoPassWatershed on lsf cluster
    """
    pass


#
# Implementation
#


def _apply_watershed_with_seeds(input_, dt, initial_seeds, config, mask, offset):
    apply_2d = config.get('apply_ws_2d', True)
    size_filter = config.get('size_filter', 25)
    sigma_weights = config.get('sigma_weights', 2.)
    alpha = config.get('alpha', 0.8)

    # apply the watersheds in 2d
    if apply_2d:
        ws = np.zeros_like(input_, dtype='uint64')
        for z in range(ws.shape[0]):

            dtz = dt[z]
            # get the initial seeds for this slice
            # and a mask for the inital seeds
            initial_seeds_z = initial_seeds[z]
            initial_seed_mask = initial_seeds_z != 0
            # don't place maxima at initial seeds
            dtz[initial_seed_mask] = 0

            seeds = _make_seeds(dtz, config)
            # remove seeds in mask
            if mask is not None:
                seeds[mask[z]] = 0

            # add offset to seeds
            seeds[seeds != 0] += offset
            # add initial seeds
            seeds[initial_seed_mask] = initial_seeds_z[initial_seed_mask]

            # we need to remap the seeds consecutively, because vigra
            # watersheds can only handle uint32 seeds, and we WILL overflow uint32
            seeds, _, old_to_new = vigra.analysis.relabelConsecutive(seeds,
                                                                     start_label=1,
                                                                     keep_zeros=True)
            new_to_old = {new: old for old, new in old_to_new.items()}

            # run watershed
            hmap = _make_hmap(input_[z], dtz, alpha, sigma_weights)
            wsz, max_id = vu.watershed(hmap, seeds=seeds, size_filter=size_filter,
                                       exclude=initial_seeds_z)
            # mask the result if we have a mask
            if mask is not None:
                wsz[mask[z]] = 0
                inv_mask = np.logical_not(mask[z])
                # NOTE we might not have any pixels in mask for 2d slice
                max_id = int(wsz[inv_mask].max()) if inv_mask.sum() > 0 else 0

            # increase the offset
            offset += max_id
            # map back to original ids
            wsz = nt.takeDict(new_to_old, wsz)
            ws[z] = wsz
        #
        return ws

    # apply the watersheds in 3d
    else:
        # find seeds
        seeds = _make_seeds(dt, config)
        # remove seeds in mask
        if mask is not None:
            seeds[mask] = 0
        seeds[seeds != 0] += offset

        # add the initial seeds
        initial_seed_mask = initial_seeds != 0
        seeds[initial_seed_mask] = initial_seeds[initial_seed_mask]

        # we need to remap the seeds consecutively, because vigra
        # watersheds can only handle uint32 seeds, and we WILL overflow uint32
        seeds, _, old_to_new = vigra.analysis.relabelConsecutive(seeds,
                                                                 start_label=1,
                                                                 keep_zeros=True)
        new_to_old = {new: old for old, new in old_to_new.items()}

        # run watershed
        initial_seed_ids = np.unique(initial_seeds[initial_seed_mask])
        hmap = _make_hmap(input_, dt, alpha, sigma_weights)
        ws, max_id = vu.watershed(hmap, seeds=seeds, size_filter=size_filter,
                                  exclude=initial_seed_ids)
        ws = nt.takeDict(new_to_old, ws)
        if mask is not None:
            ws[mask] = 0
        return ws


def _ws_pass2(blocking, block_id, ds_in, ds_out, mask, config):
    fu.log("start processing block %i" % block_id)

    input_bb, inner_bb, output_bb = _get_bbs(blocking, block_id, config)
    # get the mask and check if we have any pixels
    if mask is None:
        in_mask = out_mask = None
    else:
        in_mask = mask[input_bb].astype('bool')
        out_mask = in_mask[inner_bb]
        if np.sum(out_mask) == 0:
            fu.log_block_success(block_id)
            return

    # read the input
    input_ = _read_data(ds_in, input_bb, config)

    # load seeds
    initial_seeds = ds_out[input_bb]

    if mask is None:
        inv_mask = None
    else:
        # mask the input
        inv_mask = np.logical_not(in_mask)
        input_[inv_mask] = 1

    # apply distance transform
    dt = _apply_dt(input_, config)
    # check if input was valid
    if dt is None:
        fu.log_block_success(block_id)
        return

    # get offset to make new seeds unique between blocks
    # (we need to relabel later to make processing efficient !)
    offset = block_id * np.prod(blocking.blockShape)

    # run watershed with initial seeds
    ws = _apply_watershed_with_seeds(input_, dt, initial_seeds, config, inv_mask, offset)

    # -> apply ws and write the results to the inner volume
    ds_out[output_bb] = ws[inner_bb]

    # log block success
    fu.log_block_success(block_id)


def two_pass_watershed(job_id, config_path):
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
    pass_id = config['pass']

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        assert ds_in.ndim in (3, 4)
        ds_out = f_out[output_key]
        assert ds_out.ndim == 3

        if 'mask_path' in config:
            mask_path = config['mask_path']
            mask_key = config['mask_key']
            mask = vu.load_mask(mask_path, mask_key, shape)
        else:
            mask = None

        ws_fu = _ws_block if pass_id == 0 else _ws_pass2
        for block_id in block_list:
            ws_fu(blocking, block_id, ds_in, ds_out, mask, config)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    two_pass_watershed(job_id, path)
