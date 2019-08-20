#! /usr/bin/python

import os
import sys
import json

# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
from cluster_tools.utils.numpy_utils import set_numpy_threads
set_numpy_threads(1)
import numpy as np

import luigi
import nifty.tools as nt
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from affogato.affinities import compute_affinities
from elf.wrapper.resized_volume import ResizedVolume

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

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    objects_path = luigi.Parameter()
    objects_key = luigi.Parameter()
    offsets = luigi.ListParameter(default=[[-1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]])
    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        config.update({'erode_by': 6, 'zero_objects_list': None,
                       'chunks': None, 'dilate_by': 2, 'erode_3d': True})
        return config

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        config = self.get_task_config()
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'objects_path': self.objects_path,
                       'objects_key': self.objects_key,
                       'offsets': self.offsets,
                       'block_shape': block_shape})

        shape = vu.get_shape(self.input_path, self.input_key)
        dtype = vu.file_reader(self.input_path, 'r')[self.input_key].dtype

        chunks = config['chunks']
        if chunks is None:
            chunks = vu.file_reader(self.input_path, 'r')[self.input_key].chunks
        assert all(bs % ch == 0 for bs, ch in zip(block_shape, chunks[1:]))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=tuple(shape), chunks=tuple(chunks),
                              dtype=dtype, compression='gzip')

        shape = shape[1:]
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


def dilate(inp, iterations, dilate_2d):
    assert inp.ndim == 3
    if dilate_2d:
        out = np.zeros_like(inp, dtype='float32')
        for z in range(inp.shape[0]):
            out[z] = binary_dilation(inp[z], iterations=iterations).astype('float32')
        return out
    else:
        return binary_dilation(inp, iterations=iterations).astype('float32')


def _insert_affinities(affs, objs, offsets, dilate_by):
    dtype = affs.dtype
    # compute affinities to objs and bring them to our aff convention
    affs_insert, mask = compute_affinities(objs, offsets)
    mask = mask == 0
    affs_insert = 1. - affs_insert
    affs_insert[mask] = 0

    # dilate affinity channels
    for c in range(affs_insert.shape[0]):
        affs_insert[c] = dilate(affs_insert[c], iterations=dilate_by, dilate_2d=True)
    # dirty hack: z affinities look pretty weird, so we add the averaged xy affinities
    affs_insert[0] += np.mean(affs_insert[1:3], axis=0)

    # insert affinities
    affs = vu.normalize(affs)
    affs += affs_insert
    affs = np.clip(affs, 0., 1.)
    affs = cast(affs, dtype)
    return affs


def _insert_affinities_block(block_id, blocking, ds_in, ds_out, objects, offsets,
                             erode_by, erode_3d, zero_objects_list, dilate_by):
    fu.log("start processing block %i" % block_id)
    halo = np.max(np.abs(offsets), axis=0).tolist()
    if erode_3d:
        halo = [max(ha, erode_by)
                for axis, ha in enumerate(halo)]
    else:
        halo = [ha if axis == 0 else max(ha, erode_by)
                for axis, ha in enumerate(halo)]

    block = blocking.getBlockWithHalo(block_id, halo)
    outer_bb = vu.block_to_bb(block.outerBlock)
    inner_bb = (slice(None),) + vu.block_to_bb(block.innerBlock)
    local_bb = (slice(None),) + vu.block_to_bb(block.innerBlockLocal)

    # load objects and check if we have any in this block
    # catch run-time error for singleton dimension
    try:
        objs = objects[outer_bb]
        obj_sum = objs.sum()
    except RuntimeError:
        obj_sum = 0

    # if we don't have objs, just copy the affinities
    if obj_sum == 0:
        ds_out[inner_bb] = ds_in[inner_bb]
        fu.log_block_success(block_id)
        return

    outer_bb = (slice(None),) + outer_bb
    affs = ds_in[outer_bb]

    # fit object to hmap derived from affinities via shrinking and watershed
    if erode_by > 0:
        objs, obj_ids = vu.fit_to_hmap(objs, affs[0].copy(), erode_by, erode_3d)
    else:
        obj_ids = np.unique(objs)
        if 0 in obj_ids:
            obj_ids = obj_ids[1:]

    # insert affinities to objs into the original affinities
    affs = _insert_affinities(affs, objs.astype('uint64'), offsets, dilate_by)

    # zero out some affs if necessary
    if zero_objects_list is not None:
        zero_ids = obj_ids[np.in1d(obj_ids, zero_objects_list)]
        if zero_ids.size:
            for zero_id in zero_ids:
                # erode the mask to avoid ugly boundary artifacts
                zero_mask = binary_erosion(objs == zero_id, iterations=4)
                affs[:, zero_mask] = 0

    ds_out[inner_bb] = affs[local_bb]
    fu.log_block_success(block_id)


def insert_affinities(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    objects_path = config['objects_path']
    objects_key = config['objects_key']

    erode_by = config['erode_by']
    erode_3d = config.get('erode_3d', True)
    zero_objects_list = config['zero_objects_list']
    dilate_by = config.get('dilate_by', 2)

    fu.log("Fitting objects to affinities with erosion strenght %i and erosion in 3d: %s" % (erode_by, str(erode_3d)))
    if zero_objects_list is not None:
        fu.log("Zeroing affinities for the objects %s" % str(zero_objects_list))

    block_list = config['block_list']
    block_shape = config['block_shape']
    offsets = config['offsets']

    with vu.file_reader(input_path) as f_in, vu.file_reader(output_path) as f_out,\
            vu.file_reader(objects_path) as f_obj:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]
        shape = ds_in.shape[1:]

        # TODO actually check that objects are on a lower scale
        ds_objs = f_obj[objects_key]
        objects = ResizedVolume(ds_objs, shape)

        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)
        [_insert_affinities_block(block_id, blocking, ds_in, ds_out, objects, offsets,
                                  erode_by, erode_3d, zero_objects_list, dilate_by)
         for block_id in block_list]

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    insert_affinities(job_id, path)
