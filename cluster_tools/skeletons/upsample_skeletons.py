#! /bin/python

import os
import sys
import json
from functools import partial
from concurrent import futures
from math import ceil

import numpy as np
import luigi
import nifty
import nifty.tools as nt
from scipy.ndimage.morphology import distance_transform_edt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# upsample_skeletons tasks
#

class UpsampleSkeletonsBase(luigi.Task):
    """ UpsampleSkeletons base class
    """

    task_name = 'upsample_skeletons'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    skeleton_path = luigi.Parameter()
    skeleton_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'halo': None, 'pixel_pitch': None})
        return config

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape

        # load the upsample_skeletons config
        task_config = self.get_task_config()

        # require output dataset
        chunks = (25, 256, 256)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'skeleton_path': self.skeleton_path, 'skeleton_key': self.skeleton_key,
                            'output_path': self.output_path, 'output_key': self.output_key})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
            self._write_log("scheduled %i blocks to run" % len(block_list))
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class UpsampleSkeletonsLocal(UpsampleSkeletonsBase, LocalTask):
    """
    upsample_skeletons on local machine
    """
    pass


class UpsampleSkeletonsSlurm(UpsampleSkeletonsBase, SlurmTask):
    """
    upsample_skeletons on slurm cluster
    """
    pass


class UpsampleSkeletonsLSF(UpsampleSkeletonsBase, LSFTask):
    """
    upsample_skeletons on lsf cluster
    """
    pass


#
# Implementation
#

def _upsample_skeleton(skel_id, seg, skels, scale_factor):
    # get the (inverse) mask
    mask = seg != skel_id
    # get the distances and set to max outside of the object
    distances = distance_transform_edt(mask, sampling=pixel_pitch)
    distances[mask] = np.max(distances)
    # TODO upsample the skeleton ids from skels
    coords = np.where(skels == skel_id)
    coords = tuple(coord * scale for coord, scale in zip(coords, scale_factor))
    out = np.zeros_like(distances, dtype='')
    out[coords] = ''
    # TODO connect points via shortest path on distances
    g = nifty.graph.undirectedGridGraph(distances.shape)
    edge_map = g.imageToEdgeMap(distances, 'interpixel')
    return out


def _upsample_block(block_id, blocking, halo,
                    ds_in, ds_out, ds_skel,
                    scale_factor, pixel_pitch):
    fu.log("start processing block %i" % block_id)
    if halo is None:
        block = blocking.getBlock(block_id)
        inner_bb = outer_bb = vu.block_to_bb(block)
        local_bb = np.s_[:]
    else:
        block = blocking.getBlockWithHalo(block_id, halo)
        inner_bb = vu.block_to_bb(block.innerBlock)
        outer_bb = vu.block_to_bb(block.outerBlock)
        local_bb = vu.block_to_bb(block.innerBlockLocal)

    # load the segmentation
    seg = ds_in[outer_bb]
    skels_out = np.zeros_like(seg, dtype='uint64')

    # find the bounding box for downsampled skeletons
    skel_bb = tuple(slice(b.start // scale,
                          int(ceil(b.stop / scale)))
                    for b, scale in zip(outer_bb, scale_factor))
    skels = ds_skel[skel_bb]

    # get ids of skeletons in this block (excluding zeros)
    ids = np.unique(skels)[1:]
    for skel_id in ids:
        upsampled_skel = _upsample_skeleton(skel_id, seg,
                                            skels, scale_factor)
        skels_out += upsampled_skel

    ds_skel[inner_bb] = skels_out[local_bb]
    # log block success
    fu.log_block_success(block_id)


def upsample_skeletons(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']

    output_path = config['output_path']
    output_key = config['output_key']

    skeleton_path = config['skeleton_path']
    skeleton_key = config['skeleton_key']

    block_list = config['block_list']
    halo = config.get('halo', None)
    pixel_pitch = config.get('pixel_pitch', None)

    # load input segmentation
    with vu.file_reader(input_path) as f_in, vu.file_reader(skeleton_path) as f_skel:
        shape = f[input_key].shape
        skel_shape = f[skeleton_key].shape

    scale_factor = tuple(sh // sksh for sh, sksh in zip(shape, skel_shape))
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    with vu.file_reader(input_path) as f_in,\
         vu.file_reader(skeleton_path) as f_skel,\
         vu.file_reader(output_path) as f_out:

        [_upsample_block(block_id, blocking, halo,
                         ds_in, ds_out, ds_skel,
                         scale_factor, pixel_pitch)
        for block_id in block_list]

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    upsample_skeletons(job_id, path)
