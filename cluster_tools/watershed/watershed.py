#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import vigra
import nifty.tools as nt

# TODO which task do we need
import cluster_tools.volume_util as vu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask
from cluster_tools.functional_api import log_job_success, log_block_success, log, load_global_config


#
# Watershed Tasks
#

# TODO there is still some code copy for differnt implementations (local, slurm, lsf)
# would be nice to avoid this
class WatershedLocal(LocalTask):
    task_name = 'watershed'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    # configuration paths
    global_config_path = luigi.Parameter()
    ws_config_path = luigi.Parameter()

    def run(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = load_global_config(self.global_config_path)
        self.shebang = shebang
        self.init()

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)
        if len(shape) == 4:
            shape = shape[1:]
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # load the watershed config
        # TODO check more parameters here
        with open(self.ws_config_path, 'r') as f:
            ws_config = json.load(f)
        # update the config with input and output paths and keys
        # as well as block shape
        ws_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                          'output_path': self.output_path, 'output_key': self.output_key,
                          'block_shape': block_shape})

        # require output dataset
        # TODO read chunks from config
        chunks = tuple(bs // 2 for bs in block_shape)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, **ws_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish (actually not necessary for local task)
        # and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


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
def _apply_watershed(input_, dt, config):
    apply_2d = config.get('apply_ws_2d', False)
    sigma_seeds = config.get('sigma_seeds', 0.)
    size_filter = config.get('size_filter', 0)

    # apply the watersheds in 2d
    if apply_2d:
        ws = np.zeros_like(input_, dtype='uint64')
        offset = 0
        for z in range(ws.shape[0]):
            dtz = vu.apply_filter(dt[z], 'gaussianSmoothing',
                               sigma_seeds) if sigma_seeds > 0 else dt[z]
            seeds = vigra.analysis.localMaxima(dtz, marker=np.nan,
                                               allowAtBorder=True, allowPlateaus=True)
            seeds = vigra.analysis.labelImageWithBackground(np.isnan(seeds).view('uint8'))
            wsz, max_id = vigra.analysis.watershedsNew(input_[z], seeds=seeds)
            # apply size_filter if specified
            if size_filter > 0:
                wsz, max_id = vu.apply_size_filter(wsz, input_[z], size_filter)
            wsz += offset
            ws[z] = wsz
            offset += max_id

    # apply the watersheds in 3d
    else:
        seeds = vigra.analysis.localMaxima3D(dtz, marker=np.nan,
                                             allowAtBorder=True, allowPlateaus=True)
        seeds = vigra.analysis.labelVolumeWithBackground(np.isnan(seeds).view('uint8'))
        ws, max_id = vigra.analysis.watershedsNew(input_, seeds=seeds)
        # apply size_filter if specified
        if size_filter > 0:
            ws, max_id = vu.apply_size_filter(ws, input_, size_filter)
    #
    return ws, max_id


# TODO with masks
def _ws_block(blocking, block_id, ds_in, ds_out, config):
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
        # TODO enable reading only subset of channels
        input_bb = (slice(None),) + input_bb
        input_ = vu.normalize(ds_in[input_bb])
        # TODO allow for different agglomeration funtions
        # (mean, max, min) and channel weights for averaging
        input_ = np.mean(input_, axis=0)
    else:
        input_ = vu.normalize(ds_in[input_bb])

    # smooth input if sigma is given
    sigma_weights = float(config.get('sigma_weights', 0.))
    if sigma_weights > 0:
        input_ = vu.apply_filter(input_, 'gaussianSmoothing', sigma)

    # apply distance transform
    dt = _apply_dt(input_, config)

    # TODO add option to load prev watershed results as seeds
    # in this case we also need to add the option to increase the output bb
    # apply watershed
    ws, max_id = _apply_watershed(input_, dt, config)

    # write to output
    ds_out[output_bb] = ws[inner_bb]

    # TODO how do we store the max-id in this block ?
    # log block success
    log_block_success(block_id)


def watershed(job_id, config_path):
    log("start processing job %i" % job_id)
    log("reading config from %s" % config_path)
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
        ds_in  = f_in[input_key]
        assert ds_in.ndim in (3, 4)
        ds_out = f_out[output_key]
        assert ds_out.ndim == 3
        for block_id in block_list:
            _ws_block(blocking, block_id, ds_in, ds_out, config)
    # log success
    log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    watershed(job_id, path)
