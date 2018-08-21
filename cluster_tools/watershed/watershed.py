#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import vigra
import nifty.tools as nt

import cluster_tools.volume_util as vu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.functional_api import log_job_success, log_block_success, log, load_global_config


#
# Watershed Tasks
#

# TODO implement watershed with mask
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

    # configuration paths
    ws_config_path = luigi.Parameter()

    def _watershed_pass(self, n_jobs, block_list, ws_config, prefix=None):
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, ws_config, prefix)
        self.submit_jobs(n_jobs, prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(prefix)
        self.check_jobs(n_jobs, prefix)

    def run(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = load_global_config(self.global_config_path)
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)
        if len(shape) == 4:
            shape = shape[1:]

        # load the watershed config
        # TODO check more parameters here
        with open(self.ws_config_path, 'r') as f:
            ws_config = json.load(f)

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

        # check if we run a 2-pass watershed
        is_2pass = ws_config.pop('is_2pass', False)

        # run 2 passes of watersheds with checkerboard pattern
        # for the blocks
        if is_2pass:
            assert 'halo' in ws_config, "Need halo for two-pass wlatershed"
            blocking = nt.blocking([0, 0, 0], list(shape), list(block_shape))
            blocks_1, blocks_2 = vu.make_checkerboard_block_lists(blocking, roi_begin, roi_end)
            n_jobs = min(len(blocks_1), self.max_jobs)
            ws_config['pass'] = 1
            self._watershed_pass(n_jobs, blocks_1, ws_config, 'pass1')
            ws_config['pass'] = 2
            self._watershed_pass(n_jobs, blocks_2, ws_config, 'pass2')
        # run single pass watershed with all blocks in block_list
        else:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
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

    pixel_pitch = config.get('pixel_pitch', None)
    apply_2d = config.get('apply_dt_2d', False)
    if apply_2d:
        assert pixel_pitch is None
        dt = np.zeros_like(threshd, dtype='float32')
        for z in range(dt.shape[0]):
            dt[z] = vigra.filters.distanceTransform(threshd[z])

    else:
        dt = vigra.filters.distanceTransform(threshd) if pixel_pitch is None else\
            vigra.filters.distanceTransform(thresd, pixel_pitch=pixel_pitch)

    return dt


# apply watershed
def _apply_watershed(input_, dt, offset, config):
    apply_2d = config.get('apply_ws_2d', False)
    sigma_seeds = config.get('sigma_seeds', 0.)
    size_filter = config.get('size_filter', 0)

    # apply the watersheds in 2d
    if apply_2d:
        ws = np.zeros_like(input_, dtype='uint64')
        for z in range(ws.shape[0]):
            # compute seeds for this slice
            dtz = vu.apply_filter(dt[z], 'gaussianSmoothing',
                                  sigma_seeds) if sigma_seeds != 0 else dt[z]
            seeds = vigra.analysis.localMaxima(dtz, marker=np.nan,
                                               allowAtBorder=True, allowPlateaus=True)
            seeds = vigra.analysis.labelImageWithBackground(np.isnan(seeds).view('uint8'))
            # run watershed for this slize
            wsz, max_id = vigra.analysis.watershedsNew(input_[z], seeds=seeds)
            # apply size_filter if specified
            if size_filter > 0:
                wsz, max_id = vu.apply_size_filter(wsz, input_[z], size_filter)
            wsz += offset
            ws[z] = wsz
            offset += max_id

    # apply the watersheds in 3d
    else:
        if sigma_seeds != 0:
            dt = vu.apply_filter(dt, 'gaussianSmoothing', sigma_seeds)
        seeds = vigra.analysis.localMaxima3D(dt, marker=np.nan,
                                             allowAtBorder=True, allowPlateaus=True)
        seeds = vigra.analysis.labelVolumeWithBackground(np.isnan(seeds).view('uint8'))
        ws, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds)
        # apply size_filter if specified
        if size_filter > 0:
            ws, max_id = vu.apply_size_filter(ws, input_, size_filter)
        ws = ws.astype('uint64')
        ws += offset
    #
    return ws


def _apply_watershed_with_seeds(input_, dt, offset, initial_seeds, config):
    apply_2d = config.get('apply_ws_2d', False)
    sigma_seeds = config.get('sigma_seeds', 0.)
    size_filter = config.get('size_filter', 0)

    # apply the watersheds in 2d
    if apply_2d:
        ws = np.zeros_like(input_, dtype='uint64')
        for z in range(ws.shape[0]):
            # get seesds in this slice
            dtz = vu.apply_filter(dt[z], 'gaussianSmoothing',
                                  sigma_seeds) if sigma_seeds != 0 else dt[z]
            seeds = vigra.analysis.localMaxima(dtz, marker=np.nan,
                                               allowAtBorder=True, allowPlateaus=True)
            seeds = vigra.analysis.labelImageWithBackground(np.isnan(seeds).view('uint8'))

            # add offset to seeds
            seeds[seeds != 0] += offset

            # add the initial seeds where we have them
            initial_seeds_z = initial_seeds[z]
            initial_seed_mask = initial_seeds_z != 0
            seeds[initial_seed_mask] = initial_seeds_z[initial_seed_mask]

            # we need to remap the seeds consecutively, because vigra
            # watersheds can only handle uint32 seeds, and we WILL overflow uint32
            seeds, _, old_to_new = vigra.analysis.relabelConsecutive(seeds,
                                                                     start_label=1,
                                                                     keep_zeros=True)
            new_to_old = {new: old for old, new in old_to_new.items()}

            # run watershed
            _, max_id = vigra.analysis.watershedsNew(input_[z], seeds=seeds, out=seeds)

            # apply size_filter if specified
            if size_filter > 0:
                # we do not filter ids from the initial seed mask
                initial_seed_ids = np.unique(initial_seeds_z[initial_seed_mask])
                seeds, max_id = vu.apply_size_filter(seeds, input_[z], size_filter, exclude=initial_seed_ids)

            # map back to original ids
            seeds = seeds.astype('uint64')
            seeds = nt.takeDict(new_to_old, seeds)
            ws[z] = seeds
            offset += max_id
        return ws

    # apply the watersheds in 3d
    else:
        if sigma_seeds != 0:
            dt = vu.apply_filter(dt, 'gaussianSmoothing', sigma_seeds)

        # find seeds
        seeds = vigra.analysis.localMaxima3D(dt, marker=np.nan,
                                             allowAtBorder=True, allowPlateaus=True)
        seeds = vigra.analysis.labelVolumeWithBackground(np.isnan(seeds).view('uint8'))
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
        _, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds, out=seeds)

        # apply size_filter if specified
        if size_filter > 0:
            # we do not filter ids from the initial seed mask
            initial_seed_ids = np.unique(initial_seeds_z[initial_seed_mask])
            seeds, max_id = vu.apply_size_filter(seeds, input_[z], size_filter, exclude=initial_seed_ids)
        seeds = seeds.astype('uint64')
        seeds = nt.takeDict(new_to_old, seeds)
        return seeds


def _ws_block(blocking, block_id, ds_in, ds_out, config, pass_):
    log("start processing block %i" % block_id)

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

    # read the input data
    if ds_in.ndim == 4:
        channel_end = config.get('channel_end', None)
        if channel_end is None:
            input_bb = (slice(0, channel_end),) + input_bb
        else:
            input_bb = (slice(None),) + input_bb
        input_ = vu.normalize(ds_in[input_bb])
        agglomerate = config.get('agglomerate_channels', 'mean')
        assert agglomerate in ('mean', 'max', 'min')
        input_ = getattr(np, agglomerate)(input_, axis=0)
    else:
        input_ = vu.normalize(ds_in[input_bb])

    # smooth input if sigma is given
    sigma_weights = float(config.get('sigma_weights', 0.))
    if sigma_weights > 0:
        input_ = vu.apply_filter(input_, 'gaussianSmoothing', sigma)

    # apply distance transform
    dt = _apply_dt(input_, config)

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
        ws = _apply_watershed_with_seeds(input_, dt, offset, initial_seeds, config)
        ds_out[output_bb] = ws[inner_bb]

    # log block success
    log_block_success(block_id)


def watershed(job_id, config_path):
    log("start processing job %i" % job_id)
    log("reading config from %s" % config_path)
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

    # get the blocking
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in  = f_in[input_key]
        assert ds_in.ndim in (3, 4)
        ds_out = f_out[output_key]
        assert ds_out.ndim == 3
        for block_id in block_list:
            _ws_block(blocking, block_id, ds_in, ds_out, config, pass_)
    # log success
    log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    watershed(job_id, path)
