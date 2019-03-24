#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import vigra
from scipy.ndimage.morphology import binary_erosion
import nifty.tools as nt
from affogato.affinities import compute_affinities

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Insert affs
#

class InsertAffinitiesBase(luigi.Task):
    """ InsertAffinities base class
    """

    task_name = 'insert_affinities'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    affinity_path = luigi.Parameter()
    affinity_key = luigi.Parameter()
    objects_path = luigi.Parameter()
    objects_key = luigi.Parameter()
    offsets = luigi.ListParameter(default=[[-1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]])
    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        config.update({'erode_by': 10, 'zero_objects_list': None})
        return config

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        config = self.get_task_config()
        config.update({'affinity_path': self.affinity_path,
                       'affinity_key': self.affinity_key,
                       'objects_path': self.objects_path,
                       'objects_key': self.objects_key,
                       'offsets': self.offsets,
                       'block_shape': block_shape})

        shape = vu.get_shape(self.affinity_path, self.affinity_key)[1:]
        chunks = vu.file_reader(self.affinity_path)[self.affinity_key].chunks[1:]
        assert all(bs % ch == 0 for bs, ch in zip(block_shape, chunks))

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class InsertAffinitiesLocal(InsertAffinitiesBase, LocalTask):
    """
    InsertAffinities on local machine
    """
    pass


class InsertAffinitiesSlurm(InsertAffinitiesBase, SlurmTask):
    """
    InsertAffinities on slurm cluster
    """
    pass


class InsertAffinitiesLSF(InsertAffinitiesBase, LSFTask):
    """
    InsertAffinities on lsf cluster
    """
    pass


def cast(input_, dtype):
    if np.dtype(input_.dtype) == np.dtype(dtype):
        return input_
    assert dtype == np.dtype('uint8')
    input_ *= 255.
    return input_.astype('uint8')


def fit_to_hmap(objs, hmap, erode_by):
    obj_ids = np.unique(objs)
    if 0 in obj_ids:
        obj_ids = obj_ids[1:]
    bg_id = obj_ids[-1] + 1

    background = objs == 0
    seeds = bg_id * binary_erosion(background, iterations=erode_by)
    seeds = seeds.astype('uint32')

    for obj_id in obj_ids:
        obj_seeds = binary_erosion(objs == obj_id, iterations=erode_by)
        seeds[obj_seeds] = obj_id

    # apply dt and smooth hmap before watershed
    hmap = vu.normalize(hmap)
    threshold = .3
    threshd = (hmap > threshold).astype('uint32')
    dt = vigra.filters.distanceTransform(threshd)
    dt = vu.normalize(dt)
    alpha = .75
    hmap = alpha * hmap + (1. - alpha) * dt

    objs = vigra.analysis.watershedsNew(hmap, seeds=seeds)[0]
    objs[objs == bg_id] = 0
    return objs, obj_ids


def _insert_affinities_block(block_id, blocking, ds, objects, offsets,
                             erode_by, zero_objects_list):
    fu.log("start processing block %i" % block_id)
    halo = np.max(np.abs(offsets), axis=0)

    block = blocking.getBlockWithHalo(block_id, halo.tolist())
    outer_bb = vu.block_to_bb(block.outerBlock)
    inner_bb = (slice(None),) + vu.block_to_bb(block.innerBlock)
    local_bb = (slice(None),) + vu.block_to_bb(block.innerBlockLocal)

    # load objects and check if we have any in this block
    # catch run-time error for singleton dimension
    try:
        objs = objects[outer_bb]
    except RuntimeError:
        fu.log_block_success(block_id)
        return

    if objs.sum() == 0:
        fu.log_block_success(block_id)
        return

    outer_bb = (slice(None),) + outer_bb
    # print(outer_bb)
    affs = ds[outer_bb]
    max_aff = 255 if np.dtype(affs.dtype) == np.dtype('uint8') else 1.

    # fit object to hmap derived from affinities via shrinking and watershed
    if erode_by > 0:
        objs, obj_ids = fit_to_hmap(objs, affs[0].copy(), erode_by)
    else:
        obj_ids = np.unique(objs)
        if 0 in obj_ids:
            obj_ids = obj_ids[1:]

    affs_insert, _ = compute_affinities(objs.astype('uint64'), offsets)
    affs_insert = cast(1. - affs_insert, ds.dtype)
    affs += affs_insert
    affs = np.clip(affs, 0, max_aff)

    # zero out some affs if necessary
    if zero_objects_list is not None:
        zero_ids = np.in1d(obj_ids, zero_objects_list)
        if zero_ids.size:
            for zero_id in zero_ids:
                zero_mask = objs == zero_id
                affs[:, zero_mask] = 0

    ds[inner_bb] = affs[local_bb]
    fu.log_block_success(block_id)


def insert_affinities(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    affinity_path = config['affinity_path']
    affinity_key = config['affinity_key']
    objects_path = config['objects_path']
    objects_key = config['objects_key']

    erode_by = config['erode_by']
    zero_objects_list = config['zero_objects_list']

    block_list = config['block_list']
    block_shape = config['block_shape']
    offsets = config['offsets']

    with vu.file_reader(affinity_path) as f_in, vu.file_reader(objects_path) as f_obj:
        ds = f_in[affinity_key]
        shape = ds.shape[1:]

        # TODO actually check that objects are on a lower scale
        ds_objs = f_obj[objects_key]
        objects = vu.InterpolatedVolume(ds_objs, shape)

        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)
        [_insert_affinities_block(block_id, blocking, ds, objects, offsets,
                                  erode_by, zero_objects_list)
         for block_id in block_list]

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    insert_affinities(job_id, path)
