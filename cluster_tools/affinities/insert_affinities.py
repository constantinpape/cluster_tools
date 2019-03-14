#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
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
        chunks = vu.file_reader(self.affinity_path)[self.affinity_key][1:]
        assert all(bs & ch == 0 for bs, ch in zip(block_shape, chunks))

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


def _insert_affinities_block(block_id, blocking, ds, objects, offsets):
    fu.log("start processing block %i" % block_id)
    halo = np.max(np.abs(offsets), axis=0)

    block = blocking.getBlockWithHalo(block_id, halo.tolist())
    outer_bb = vu.block_to_bb(block.outerBlock)
    inner_bb = (slice(None),) + vu.block_to_bb(block.innerBlock)
    local_bb = (slice(None),) + vu.block_to_bb(block.innerBlockLocal)

    # load objects and check if we have any in this block
    objs = objects[outer_bb]
    if objs.sum() == 0:
        fu.log_block_success(block_id)
        return

    affs, _ = compute_affinities(objs, offsets)
    ds[inner_bb] = 1. - affs[local_bb]
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
        [_insert_affinities_block(block_id, blocking, ds, objects, offsets)
         for block_id in block_list]

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    insert_affinities(job_id, path)
