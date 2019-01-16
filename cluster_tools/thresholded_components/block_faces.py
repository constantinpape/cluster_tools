#! /usr/bin/python

import os
import sys
import json
import pickle

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Find Labeling Tasks
#

class BlockFacesBase(luigi.Task):
    """ BlockFaces base class
    """

    task_name = 'block_faces'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    offsets_path = luigi.Parameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        config = self.get_task_config()
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'offsets_path': self.offsets_path,
                       'block_shape': block_shape,
                       'tmp_folder': self.tmp_folder})

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


class BlockFacesLocal(BlockFacesBase, LocalTask):
    """
    BlockFaces on local machine
    """
    pass


class BlockFacesSlurm(BlockFacesBase, SlurmTask):
    """
    BlockFaces on slurm cluster
    """
    pass


class BlockFacesLSF(BlockFacesBase, LSFTask):
    """
    BlockFaces on lsf cluster
    """
    pass


def _process_face(blocking, block_id,
                  axis, ds, offsets,
                  empty_blocks):
    ngb_id = blocking.getNeighborId(block_id, axis, False)
    if ngb_id == -1 or ngb_id in empty_blocks:
        return None
    assert ngb_id > block_id, "%i, %i" % (ngb_id, block_id)

    off_a = offsets[block_id]
    off_b = offsets[ngb_id]

    block_a = blocking.getBlock(block_id)
    block_b = blocking.getBlock(ngb_id)
    assert all(beg_a == beg_b for dim, beg_a, beg_b
               in zip(range(3), block_a.begin, block_b.begin) if dim != axis)
    assert all(end_a == end_b for dim, end_a, end_b
               in zip(range(3), block_a.end, block_b.end) if dim != axis)
    assert block_a.begin[axis] < block_b.begin[axis]

    # compute the bounding box corresponiding to the face between the two blocks
    face = tuple(slice(beg, end) if dim != axis else slice(end - 1, end + 1)
                 for dim, beg, end in zip(range(3), block_a.begin, block_a.end))

    seg = ds[face]
    assert seg.shape[axis] == 2, "%i: %s" % (axis, str(seg.shape))

    # get the local coordinates of faces in a and b
    slice_a = slice(0, 1)
    slice_b = slice(1, 2)
    face_a = tuple(slice(None) if dim != axis else slice_a
                   for dim in range(3))
    face_b = tuple(slice(None) if dim != axis else slice_b
                   for dim in range(3))

    # load the local faces
    labels_a = seg[face_a].squeeze()
    labels_b = seg[face_b].squeeze()
    assert labels_a.size > 0
    assert labels_a.shape == labels_b.shape

    have_labels = np.logical_and(labels_a != 0, labels_b != 0)
    labels_a = labels_a[have_labels]
    labels_b = labels_b[have_labels]
    assert labels_a.shape == labels_b.shape

    if labels_a.size == 0:
        return None

    # add the offsets that make the block ids unique
    labels_a += off_a
    labels_b += off_b

    assignments = np.concatenate((labels_a[:, None], labels_b[:, None]), axis=1)
    assignments = np.unique(assignments, axis=0)
    return assignments


def _process_faces(block_id, blocking, ds, offsets, empty_blocks):
    fu.log("start processing block %i" % block_id)
    if block_id in empty_blocks:
        fu.log_block_success(block_id)
        return None

    assignments = [_process_face(blocking, block_id,
                                 axis, ds, offsets,
                                 empty_blocks)
                   for axis in range(3)]
    assignments = [ass for ass in assignments
                   if ass is not None]

    # all assignments might be None, so we need to check for taht
    if assignments:
        assignments = np.unique(np.concatenate(assignments, axis=0),
                                axis=0)
    else:
        assignments = None
    fu.log_block_success(block_id)
    return assignments


def block_faces(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    block_list = config['block_list']
    tmp_folder = config['tmp_folder']
    offsets_path = config['offsets_path']
    block_shape = config['block_shape']

    with open(offsets_path) as f:
        offsets_dict = json.load(f)
        offsets = offsets_dict['offsets']
        empty_blocks = offsets_dict['empty_blocks']
        n_labels = offsets_dict['n_labels']

    with vu.file_reader(input_path, 'r') as f:
        ds = f[input_key]
        shape = list(ds.shape)

        blocking = nt.blocking([0, 0, 0], shape, block_shape)
        assignments = [_process_faces(block_id, blocking, ds,
                                      offsets, empty_blocks)
                       for block_id in block_list]
    # filter out empty assignments
    assignments = [ass for ass in assignments if ass is not None]
    if assignments:
        assignments = np.concatenate(assignments, axis=0)
        assignments = np.unique(assignments, axis=0)
        assert assignments.max() < n_labels, "%i, %i" % (int(assignments.max()), n_labels)

    save_path = os.path.join(tmp_folder, 'assignments_%i.npy' % job_id)
    np.save(save_path, assignments)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    block_faces(job_id, path)