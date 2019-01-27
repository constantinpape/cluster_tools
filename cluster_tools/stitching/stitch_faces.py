#! /usr/bin/python

import os
import sys
import json
import pickle

import luigi
import numpy as np
import nifty.tools as nt
import nifty.ground_truth as ngt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Stitching Tasks
#

class StitchFacesBase(luigi.Task):
    """ StitchFaces base class
    """

    task_name = 'stitch_faces'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    shape = luigi.Parameter()
    overlap_prefix = luigi.Parameter()
    save_prefix = luigi.Parameter()
    offsets_path = luigi.Parameter()
    overlap_threshold = luigi.FloatParameter()
    halo = luigi.ListParameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        block_list = vu.blocks_in_volume(self.shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        config = self.get_task_config()
        config.update({'shape': self.shape,
                       'offsets_path': self.offsets_path,
                       'overlap_prefix': self.overlap_prefix,
                       'save_prefix': self.save_prefix,
                       'overlap_threshold': self.overlap_threshold,
                       'block_shape': block_shape,
                       'tmp_folder': self.tmp_folder,
                       'halo': self.halo})

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class StitchFacesLocal(StitchFacesBase, LocalTask):
    """
    StitchFaces on local machine
    """
    pass


class StitchFacesSlurm(StitchFacesBase, SlurmTask):
    """
    StitchFaces on slurm cluster
    """
    pass


class StitchFacesLSF(StitchFacesBase, LSFTask):
    """
    StitchFaces on lsf cluster
    """
    pass


def _stitch_face(offsets, overlap_prefix, block_a, block_b,
                 face_a, face_b, overlap_threshold):
    off_a = offsets[block_a]
    off_b = offsets[block_b]

    path_a = '%s_%i_%i.npy' % (overlap_prefix, block_a, block_b)
    path_b = '%s_%i_%i.npy' % (overlap_prefix, block_b, block_a)

    # overlaps might not exist because they are empty
    if (not os.path.exists(path_a)) or (not os.path.exists(path_b)):
        return None

    ovlp_a = np.load(path_a)
    ovlp_b = np.load(path_b)
    assert ovlp_a.shape == ovlp_b.shape, "%s, %s" % (str(ovlp_a.shape), str(ovlp_b.shape))

    # find the block face and the ids on the block face
    axis = vu.faces_to_ovlp_axis(face_a, face_b)
    face = tuple(slice(None) if dim != axis else
                 slice(fa.stop - 1, fa.stop + 1) for dim, fa in enumerate(face_a))

    # find the ids ON the actual block boundary
    segments_a = np.unique(ovlp_a[face])
    segments_b = np.unique(ovlp_b[face])

    # 0 is ignore label, so we don't consider it here
    if segments_a[0] == 0:
        segments_a = segments_a[1:]
    if segments_b[0] == 0:
        segments_b = segments_b[1:]

    # measure all overlaps
    overlaps_ab = ngt.overlap(ovlp_a, ovlp_b)
    overlaps_ba = ngt.overlap(ovlp_b, ovlp_a)

    assignments = []
    # iterate over all segments on the face of block a
    for seg_a in segments_a:

        # get the segments of block b overlapping with seg_a
        ovlp_seg_a, counts_seg_a = overlaps_ab.overlapArraysNormalized(seg_a, sorted=True)
        seg_b = ovlp_seg_a[0]
        # continue if the max overlapping object is not on the face of block b
        if seg_b not in segments_b:
            continue

        # continue if the max overlapping objects do not agree
        ovlp_seg_b, counts_seg_b = overlaps_ba.overlapArraysNormalized(seg_b, sorted=True)
        if ovlp_seg_b[0] != seg_a:
            continue

        # merge the two ids if their mean overlap is larger than the overlap threshold
        ovlp_measure = (counts_seg_a[0] + counts_seg_b[0]) / 2.
        if ovlp_measure > overlap_threshold:
            assignments.append([seg_a + off_a, seg_b + off_b])

    if assignments:
        return np.array(assignments, dtype='uint64')
    else:
        return None


def _stitch_faces(block_id, blocking, halo,
                  overlap_prefix, overlap_threshold,
                  offsets, empty_blocks):
    fu.log("start processing block %i" % block_id)
    if block_id in empty_blocks:
        fu.log_block_success(block_id)
        return None

    assignments = [_stitch_face(offsets, overlap_prefix, block_a, block_b,
                                face_a, face_b, overlap_threshold)
                   for _, face_a, face_b, block_a, block_b
                   in vu.iterate_faces(blocking, block_id,
                                       return_only_lower=True,
                                       empty_blocks=empty_blocks,
                                       halo=halo)]
    assignments = [ass for ass in assignments if ass is not None]

    # all assignments might be None, so we need to check for that
    if assignments:
        assignments = np.concatenate(assignments, axis=0)
    else:
        assignments = None
    fu.log_block_success(block_id)
    return assignments


def stitch_faces(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    shape = config['shape']
    offsets_path = config['offsets_path']
    tmp_folder = config['tmp_folder']
    block_shape = config['block_shape']
    block_list = config['block_list']

    save_prefix = config['save_prefix']
    overlap_prefix = config['overlap_prefix']
    overlap_threshold = config['overlap_threshold']
    halo = config['halo']

    with open(offsets_path) as f:
        offsets_dict = json.load(f)
        offsets = offsets_dict['offsets']
        empty_blocks = offsets_dict['empty_blocks']
        n_labels = offsets_dict['n_labels']

    blocking = nt.blocking([0, 0, 0], shape, block_shape)
    assignments = [_stitch_faces(block_id, blocking, halo,
                                 os.path.join(tmp_folder, overlap_prefix),
                                 overlap_threshold, offsets, empty_blocks)
                   for block_id in block_list]
    # filter out empty assignments
    assignments = [ass for ass in assignments if ass is not None]
    if assignments:
        assignments = np.concatenate(assignments, axis=0)
        assignments = np.unique(assignments, axis=0)
        assert assignments.max() < n_labels, "%i, %i" % (int(assignments.max()), n_labels)

    save_path = os.path.join(tmp_folder, '%s_%i.npy' % (save_prefix, job_id))
    np.save(save_path, assignments)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    stitch_faces(job_id, path)
