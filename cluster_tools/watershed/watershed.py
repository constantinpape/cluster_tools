#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import vigra
import nifty.tools as nt
from nifty.filters import nonMaximumDistanceSuppression

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Watershed Tasks
#

class WatershedBase(luigi.Task):
    """ Watershed base class
    """

    task_name = 'watershed'
    src_file = os.path.abspath(__file__)

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
                       'two_pass': False, 'channel_begin': 0, 'channel_end': None,
                       'agglomerate_channels': 'mean', 'alpha': 0.8,
                       'invert_inputs': False, 'non_maximum_suppression': True})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def _watershed_pass(self, n_jobs, block_list, ws_config, prefix=None):
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, ws_config, prefix)
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
        if self.mask_path != '':
            assert self.mask_key != ''
            ws_config.update({'mask_path': self.mask_path, 'mask_key': self.mask_key})

        # check if we run a 2-pass watershed
        is_2pass = ws_config.pop('two_pass', False)

        # run 2 passes of watersheds with checkerboard pattern
        # for the blocks
        if is_2pass:

            assert block_list_path is None, "Can't rerun watersheds if 2-pass is activated"

            # retries for two pass watershed are too complicated now
            # this could be fixed if we seperate the passes in two different
            # task instances
            self.allow_retry = False

            assert 'halo' in ws_config, "Need halo for two-pass wlatershed"
            self._write_log("run two pass watershed")
            blocking = nt.blocking([0, 0, 0], list(shape), list(block_shape))
            blocks_1, blocks_2 = vu.make_checkerboard_block_lists(blocking, roi_begin, roi_end)
            n_jobs = min(len(blocks_1), self.max_jobs)
            ws_config['pass'] = 1
            self._watershed_pass(n_jobs, blocks_1, ws_config, 'pass1')
            ws_config['pass'] = 2
            self._watershed_pass(n_jobs, blocks_2, ws_config, 'pass2')
        # run single pass watershed with all blocks in block_list
        else:
            self._write_log("run one pass watershed")
            if self.n_retries == 0:
                block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end,
                                                 block_list_path=block_list_path)
            else:
                block_list = self.block_list
                self.clean_up_for_retry(block_list)
            self._write_log('scheduling %i blocks to be processed' % len(block_list))
            n_jobs = min(len(block_list), self.max_jobs)
            self._watershed_pass(n_jobs, block_list, ws_config)


class WatershedLocal(WatershedBase, LocalTask):
    """
    Watershed on local machine
    """
    pass


class WatershedSlurm(WatershedBase, SlurmTask):
    """
    Watershed on slurm cluster
    """
    pass


class WatershedLSF(WatershedBase, LSFTask):
    """
    Watershed on lsf cluster
    """
    pass


#
# Implementation
#

# apply the distance transform to the input
def _apply_dt(input_, config):
    # threshold the input before distance transform
    threshold = config.get('threshold', .5)
    threshd = (input_ > threshold).astype('uint32')

    # we need to check if any values were above the threshold
    if(np.sum(threshd) == 0):
        return None

    pixel_pitch = config.get('pixel_pitch', None)
    apply_2d = config.get('apply_dt_2d', True)
    if apply_2d:
        assert pixel_pitch is None
        dt = np.zeros_like(threshd, dtype='float32')
        for z in range(dt.shape[0]):
            dt[z] = vigra.filters.distanceTransform(threshd[z])

    else:
        dt = vigra.filters.distanceTransform(threshd) if pixel_pitch is None else\
            vigra.filters.distanceTransform(threshd, pixel_pitch=pixel_pitch)

    return dt


def _make_hmap(input_, distances, alpha, sigma_weights):
    distances = 1. - vu.normalize(distances)
    hmap = alpha * input_ + (1. - alpha) * distances
    # smooth input if sigma is given
    if sigma_weights != 0:
        hmap = vu.apply_filter(hmap, 'gaussianSmoothing', sigma_weights)
    return hmap


def _points_to_vol(points, shape):
    vol = np.zeros(shape, dtype='uint32')
    coords = tuple(points[:, i] for i in range(points.shape[1]))
    vol[coords] = 1
    return vigra.analysis.labelMultiArrayWithBackground(vol)


def _make_seeds(dt, config):
    sigma_seeds = config.get('sigma_seeds', 2.)
    apply_nonmax_suppression = config.get('non_maximum_suppression', True)

    # find local maxima of the distance transform
    max_fu = vigra.analysis.localMaxima if dt.ndim == 2 else vigra.analysis.localMaxima3D
    if sigma_seeds > 0:
        seeds = max_fu(vu.apply_filter(dt, 'gaussianSmoothing', sigma_seeds),
                       marker=np.nan, allowAtBorder=True, allowPlateaus=True)
    else:
        seeds = max_fu(dt, marker=np.nan, allowAtBorder=True, allowPlateaus=True)

    # check if we have just one plateau
    seeds = np.isnan(seeds)
    if np.sum(seeds) == seeds.size:
        return np.ones_like(seeds, dtype='uint32')

    # find seeds via connected components after max-suppression if enabled)
    if apply_nonmax_suppression:
        seeds = np.array(np.where(seeds)).transpose()
        seeds = nonMaximumDistanceSuppression(dt, seeds)
        seeds = _points_to_vol(seeds, dt.shape)
    else:
        seeds = vigra.analysis.labelMultiArrayWithBackground(seeds.view('uint8'))

    return seeds


# apply watershed
def _apply_watershed(input_, dt, offset, config, mask=None):
    apply_2d = config.get('apply_ws_2d', True)
    sigma_weights = config.get('sigma_weights', 2.)
    size_filter = config.get('size_filter', 25)
    alpha = config.get('alpha', 0.8)

    # apply the watersheds in 2d
    if apply_2d:
        ws = np.zeros_like(input_, dtype='uint64')
        for z in range(ws.shape[0]):
            # run watershed for this slice
            dtz = dt[z]
            seeds = _make_seeds(dtz, config)
            hmap = _make_hmap(input_[z], dtz, alpha, sigma_weights)
            wsz, max_id = vu.watershed(hmap, seeds=seeds, size_filter=size_filter)
            # mask seeds if we have a mask
            if mask is None:
                wsz += offset
            else:
                wsz[mask[z]] = 0
                inv_mask = np.logical_not(mask[z])
                # NOTE we might have no pixels in the mask for this slice
                max_id = int(wsz[inv_mask].max()) if inv_mask.sum() > 0 else 0
                wsz[inv_mask] += offset
            ws[z] = wsz
            offset += max_id

    # apply the watersheds in 3d
    else:
        seeds = _make_seeds(dt, config)
        hmap = _make_hmap(input_, dt, alpha, sigma_weights)
        ws, max_id = vu.watershed(hmap, seeds, size_filter=size_filter)
        ws += offset
        # check if we have a mask
        if mask is not None:
            ws[mask] = 0
    return ws


def _apply_watershed_with_seeds(input_, dt, offset,
                                initial_seeds, config, mask=None):
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


def _get_bbs(blocking, block_id, config):
    # read the input config
    halo = list(config.get('halo', [0, 0, 0]))
    if sum(halo) > 0:
        block = blocking.getBlockWithHalo(block_id, halo)
        input_bb = vu.block_to_bb(block.outerBlock)
        output_bb = vu.block_to_bb(block.innerBlock)
        inner_bb = vu.block_to_bb(block.innerBlockLocal)
    else:
        block = blocking.getBlock(block_id)
        input_bb = output_bb = vu.block_to_bb(block)
        inner_bb = np.s_[:]
    return input_bb, inner_bb, output_bb


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
    # check if we need to invert the input
    if config.get('invert_inputs', False):
        input_ = 1. - input_
    return input_


def _ws_block(blocking, block_id, ds_in, ds_out, config, pass_):
    fu.log("start processing block %i" % block_id)
    input_bb, inner_bb, output_bb = _get_bbs(blocking, block_id,
                                             config)
    input_ = _read_data(ds_in, input_bb, config)

    # apply distance transform
    dt = _apply_dt(input_, config)
    # check if input was valid
    if dt is None:
        fu.log_block_success(block_id)
        return

    # get offset to make new seeds unique between blocks
    # (we need to relabel later to make processing efficient !)
    offset = block_id * np.prod(blocking.blockShape)

    # check which pass we are in and apply the according watershed
    if pass_ in (1, None):
        # single-pass watershed or first pass of two-pass watershed:
        # -> apply normal ws and write the results to the inner volume
        ws = _apply_watershed(input_, dt, offset, config)
        ds_out[output_bb] = ws[inner_bb]
    else:
        # second pass of two pass watershed -> apply ws with initial seeds
        # write the results to the inner volume
        if len(input_bb) == 4:
            input_bb = input_bb[1:]
        initial_seeds = ds_out[input_bb]
        ws = _apply_watershed_with_seeds(input_, dt,
                                         offset, initial_seeds, config)
        ds_out[output_bb] = ws[inner_bb]

    # log block success
    fu.log_block_success(block_id)


def _ws_block_masked(blocking, block_id,
                     ds_in, ds_out, mask, config, pass_):
    fu.log("start processing block %i" % block_id)
    input_bb, inner_bb, output_bb = _get_bbs(blocking, block_id,
                                             config)
    # get the mask and check if we have any pixels
    in_mask = mask[input_bb].astype('bool')
    out_mask = in_mask[inner_bb]
    if np.sum(out_mask) == 0:
        fu.log_block_success(block_id)
        return
    # read the input
    input_ = _read_data(ds_in, input_bb, config)

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

    # check which pass we are in and apply the according watershed
    if pass_ in (1, None):
        # single-pass watershed or first pass of two-pass watershed:
        # -> apply normal ws and write the results to the inner volume
        ws = _apply_watershed(input_, dt, offset, config, inv_mask)
        ds_out[output_bb] = ws[inner_bb]
    else:
        # second pass of two pass watershed -> apply ws with initial seeds
        # write the results to the inner volume
        if len(input_bb) == 4:
            input_bb = input_bb[1:]
        initial_seeds = ds_out[input_bb]
        ws = _apply_watershed_with_seeds(input_, dt, offset, initial_seeds,
                                         config, inv_mask)
        ds_out[output_bb] = ws[inner_bb]

    # log block success
    fu.log_block_success(block_id)


def watershed(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # check if we are running two-pass watershed and if
    # so, if we are first or second pass
    pass_ = config.get('pass', None)
    assert pass_ in (None, 1, 2)

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

    # check if we have a mask
    with_mask = 'mask_path' in config
    if with_mask:
        mask_path = config['mask_path']
        mask_key = config['mask_key']

    # get the blocking
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        assert ds_in.ndim in (3, 4)
        ds_out = f_out[output_key]
        assert ds_out.ndim == 3

        # note that the mask is usually small enough to keep it
        # in memory (and we interpolate to get to the full volume)
        # if this does not hold need to change this code!
        if with_mask:
            mask = vu.load_mask(mask_path, mask_key, shape)
            for block_id in block_list:
                _ws_block_masked(blocking, block_id,
                                 ds_in, ds_out, mask, config, pass_)

        else:
            for block_id in block_list:
                _ws_block(blocking, block_id, ds_in, ds_out, config, pass_)
    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    watershed(job_id, path)
