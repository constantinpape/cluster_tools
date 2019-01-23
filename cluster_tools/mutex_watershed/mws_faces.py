#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.segmentation_utils as su
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Block-wise mutex watershed tasks
#

class MwsFacesBase(luigi.Task):
    """ MwsFaces base class
    """

    task_name = 'mws_faces'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    offsets = luigi.ListParameter()
    id_offsets_path = luigi.Parameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')
    dependency = luigi.TaskParameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'strides': [1, 1, 1], 'randomize_strides': False})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)
        assert len(shape) == 4, "Need 4d input for MWS"
        n_channels = shape[0]
        shape = shape[1:]

        # TODO make optional which channels to choose
        assert len(self.offsets) == n_channels,\
            "%i, %i" % (len(self.offsets), n_channels)
        assert all(len(off) == 3 for off in self.offsets)

        config = self.get_task_config()
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape, 'offsets': self.offsets,
                       'tmp_folder': self.tmp_folder, 'id_offsets_path': self.id_offsets_path})

        # check if we have a mask and add to the config if we do
        if self.mask_path != '':
            assert self.mask_key != ''
            config.update({'mask_path': self.mask_path, 'mask_key': self.mask_key})

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


class MwsFacesLocal(MwsFacesBase, LocalTask):
    """
    MwsFaces on local machine
    """
    pass


class MwsFacesSlurm(MwsFacesBase, SlurmTask):
    """
    MwsFaces on slurm cluster
    """
    pass


class MwsFacesLSF(MwsFacesBase, LSFTask):
    """
    MwsFaces on lsf cluster
    """
    pass


def _mws_face(ds_in, ds_seg, offsets, id_offsets,
              strides, randomize_strides, blocking,
              face, face_a, face_b, block_a, block_b):
    # threshold for merging regions # TODO expose as parameter
    merge_threshold = .98

    # load the affinities and segmentation for this face
    affs = ds_in[(slice(None),) + face]
    seg = ds_seg[face]

    # re-run mws from affs for the face
    # face_seg = su.mutex_watershed(affs, offsets, strides=strides,
    #                               randomize_strides=randomize_strides)
    # assert face_seg.shape == seg.shape

    # offset the two parts of the existing segmentation
    id_off_a = id_offsets[block_a]
    id_off_b = id_offsets[block_b]
    seg[face_a] += id_off_a
    seg[face_b] += id_off_b

    # get the tightended face and ids directly touching it
    axis = np.where([fa != fb for fa, fb in zip(face_a, face_b)])[0]
    assert len(axis) == 1, str(axis)
    axis = axis[0]
    face_a = tuple(fa if dim != axis else slice(fa.stop - 1, fa.stop)
                   for dim, fa in enumerate(face_a))
    face_b = tuple(fb if dim != axis else slice(fb.start, fb.start + 1)
                   for dim, fb in enumerate(face_b))

    ids_a = seg[face_a].flatten()
    ids_b = seg[face_b].flatten()
    assert ids_a.shape == ids_b.shape

    # check if touching ids should be merged according to overlap criterion
    merge_candidates = np.concatenate((ids_a[:, None], ids_b[:, None]), axis=1)
    merge_candidates = np.unique(merge_candidates, axis=0)

    assignments = []
    for merge_candidate in merge_candidates:
        id_mask = np.in1d(seg, merge_candidate).reshape(seg.shape)
        # compute the mws segmentation restricted to the two candidate ids
        # NOTE need to copy here, because mws inverts some affinities internally
        candidate_seg = su.mutex_watershed(affs.copy(), offsets, strides,
                                           randomize_strides, mask=id_mask)
        # if we have only a single new segment, or a segment that covers more
        # than the merge_threshold, merge the two candidate ids
        new_ids, new_sizes = np.unique(candidate_seg[id_mask], return_counts=True)
        if len(new_ids) == 1:
             assignments.append(merge_candidate)
             continue
        area = float(sum(new_sizes))
        ratios = [nsize / area for nsize in new_sizes]
        if any(ratio > merge_threshold):
             assignments.append(merge_candidate)

        # # check ids of this area in new face segmentation
        # face_ids, id_sizes = np.unique(face_seg[id_mask], return_counts=True)
        # # if we find only a single new id, merge
        # if len(face_ids) == 1:
        #     assignments.append(merge_candidate)
        #     continue

        # id_a, id_b = merge_candidate
        # # or if one of the ids covers more than merge threshold
        # # (hard-coded at 95 % for now), merge as well

        # # find the complete area and the new id with maximum overlap
        # area = float(sum(id_sizes))
        # max_arg = np.argmax(id_sizes)
        # max_ol_id = face_ids[max_arg]

        # # compute ratios
        # # 1.) ratio of id_mask size vs. size of max_ol_id
        # ratio_1 = face_seg_sizes[max_ol_id] / area if face_seg_sizes[max_ol_id] < area else\
        #     area / face_seg_sizes[max_ol_id]
        # # 2.) ratio restricted to face_a
        # size_a = np.sum(seg[face_a] == id_a)
        # size_1 = float(np.sum(face_seg[face_a] == max_ol_id))
        # ratio_2 = size_1 / size_a if size_1 < size_a else size_a / size_1
        # # 3.) ratio restricted to face_b
        # size_b = np.sum(seg[face_b] == id_b)
        # size_2 = float(np.sum(face_seg[face_b] == max_ol_id))
        # ratio_3 = size_2 / size_b if size_2 < size_b else size_b / size_2

        # if all(ratio > merge_threshold for ratio in (ratio_1, ratio_2, ratio_3)):
        #     assignments.append(merge_candidate)

    if assignments:
        assignments = np.array(assignments)
    else:
        assignments = None
    return assignments


def _mws_faces(block_id, blocking,
               ds_in, ds_seg,
               offsets, id_offsets,
               strides, randomize_strides,
               empty_blocks):
    fu.log("start processing block %i" % block_id)
    if block_id in empty_blocks:
        fu.log_block_success(block_id)
        return None

    # compute halo from offsets:
    # max offset + 1 in each direction
    halo = np.max(np.abs(offsets), axis=0) + 1
    assignments = [_mws_face(ds_in, ds_seg, offsets, id_offsets,
                             strides, randomize_strides, blocking,
                             face, face_a, face_b, block_a, block_b)
                   for face, face_a, face_b, block_a, block_b
                   in vu.iterate_faces(blocking, block_id,
                                       return_only_lower=True,
                                       empty_blocks=empty_blocks,
                                       halo=halo)]
    assignments = [ass for ass in assignments if ass is not None]

    # all assignments might be None, so we need to check for that
    if assignments:
        assignments = np.unique(np.concatenate(assignments, axis=0),
                                axis=0)
    else:
        assignments = None
    fu.log_block_success(block_id)
    return assignments


def _mws_faces_with_mask(block_id, blocking,
                         ds_in, ds_seg,
                         mask, offsets,
                         strides, randomize_strides):
    fu.log("start processing block %i" % block_id)
    fu.log_block_success(block_id)
    return int(seg.max()) + 1


def mws_faces(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    block_shape = config['block_shape']
    block_list = config['block_list']
    tmp_folder = config['tmp_folder']
    offsets = config['offsets']
    id_offsets_path = config['id_offsets_path']

    strides = config['strides']
    assert len(strides) == 3
    assert all(isinstance(stride, int) for stride in strides)
    randomize_strides = config['randomize_strides']
    assert isinstance(randomize_strides, bool)

    mask_path = config.get('mask_path', '')
    mask_key = config.get('mask_key', '')

    with open(id_offsets_path) as f:
        id_offsets_dict = json.load(f)
        id_offsets = id_offsets_dict['offsets']
        empty_blocks = id_offsets_dict['empty_blocks']
        n_labels = id_offsets_dict['n_labels']

    with vu.file_reader(input_path, 'r') as f_in,\
        vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_seg = f_out[output_key]

        shape = ds_in.shape[1:]
        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)

        if mask_path != '':
            assert False, "TODO"
            # note that the mask is usually small enough to keep it
            # in memory (and we interpolate to get to the full volume)
            # if this does not hold need to change this code!
            mask = vu.load_mask(mask_path, mask_key, shape)
            assignments = [_mws_faces_with_mask(block_id, blocking,
                                                ds_in, ds_seg, mask,
                                                offsets, id_offsets,
                                                strides, randomize_strides,
                                                empty_blocks)
                           for block_id in block_list]
        else:
            assignments = [_mws_faces(block_id, blocking,
                                      ds_in, ds_seg,
                                      offsets, id_offsets,
                                      strides, randomize_strides,
                                      empty_blocks)
                           for block_id in block_list]

    # filter out empty assignments
    assignments = [ass for ass in assignments if ass is not None]
    if assignments:
        assignments = np.concatenate(assignments, axis=0)
        assignments = np.unique(assignments, axis=0)
        assert assignments.max() < n_labels, "%i, %i" % (int(assignments.max()), n_labels)

    save_path = os.path.join(tmp_folder, 'mws_assignments_%i.npy' % job_id)
    np.save(save_path, assignments)
    fu.log("saving mws face assignments to %s" % save_path)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    mws_faces(job_id, path)
