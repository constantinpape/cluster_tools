#! /bin/python

import os
import sys
import json

import luigi
import vigra
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class CorrectAnchorsBase(luigi.Task):
    """ CorrectAnchors base class
    """

    task_name = 'correct_anchors'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    morphology_path = luigi.Parameter()
    morphology_key = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        number_of_labels = int(vu.file_reader(self.input_path, 'r')[self.input_key].attrs['maxId']) + 1
        id_chunks = 1000
        block_list = vu.blocks_in_volume([number_of_labels], [id_chunks])

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'morphology_path': self.morphology_path,
                       'morphology_key': self.morphology_key,
                       'id_chunks': id_chunks,
                       'tmp_folder': self.tmp_folder})

        # prime and run the jobs
        self.prepare_jobs(self.max_jobs, block_list, config)
        self.submit_jobs(self.max_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(self.max_jobs)


class CorrectAnchorsLocal(CorrectAnchorsBase, LocalTask):
    """ CorrectAnchors on local machine
    """
    pass


class CorrectAnchorsSlurm(CorrectAnchorsBase, SlurmTask):
    """ CorrectAnchors on slurm cluster
    """
    pass


class CorrectAnchorsLSF(CorrectAnchorsBase, LSFTask):
    """ CorrectAnchors on lsf cluster
    """
    pass


#
# Implementation
#


def _centers_to_chunks(blocking, labels, com, bb_start, bb_stop):
    chunk_mapping = {}
    for label, center, start, stop in zip(labels, com, bb_start, bb_stop):
        chunk_id = blocking.coordinatesToBlockId([int(ce) for ce in center])
        if chunk_id in chunk_mapping:
            chunk_mapping[chunk_id].append((label, center, bb_start, bb_stop))
        else:
            chunk_mapping[chunk_id] = [(label, center, bb_start, bb_stop)]
    return chunk_mapping


def _find_center_in_block(seg, label_id, offset):
    id_mask = seg == label_id
    if id_mask.sum() == 0:
        return None
    center = vigra.filters.eccentricityCenters(id_mask.astype('uint32'))[1]
    center = tuple(ce + off for ce, off in zip(center, offset))
    return center


def _correct_anchor_for_label(ds, blocking, label_id, bb_start, bb_stop):
    chunk_ids = blocking.getBlockIdsOverlappingBoundingBox(list(bb_start),
                                                           list(bb_stop))
    for chunk_id in chunk_ids:
        block = blocking.getBlock(chunk_id)
        offset = tuple(beg for beg in block.begin)
        bb = vu.block_to_bb(block)
        seg = ds[bb]

        anchor = _find_center_in_block(seg, label_id, offset)
        if anchor is not None:
            break

    return anchor


def _correct_anchors_for_chunk(ds, blocking, chunk_id, label_info):
    block = blocking.getBlock(chunk_id)
    offset = tuple(beg for beg in block.begin)
    bb = vu.block_to_bb(block)
    seg = ds[bb]

    corrections = {}
    for info in label_info:
        label_id, center, bb_start, bb_stop = info
        center_local = tuple(int(ce - off) for ce, off in zip(center, offset))
        center_id = seg[center_local]

        if center_id != label_id:
            continue

        # check if this id is in the current block and update the
        # anchor within the current block if it is
        anchor = _find_center_in_block(seg, label_id, offset)

        # otherwise, we need to start loading more chunks and update the
        # anchor from those
        if anchor is None:
            anchor = _correct_anchor_for_label(ds, blocking, label_id, bb_start, bb_stop)
            assert anchor is not None
        corrections[label_id] = anchor

    return corrections


def _correct_anchors_for_label_range(ds, blocking,
                                     label_ids, com, bb_start, bb_stop,
                                     label_begin, label_end):

    # get com etc. for the current ids
    this_ids = np.arange(label_begin, label_end)
    id_mask = np.in1d(label_ids, this_ids)
    this_ids = label_ids[id_mask]
    this_com = com[id_mask]
    this_bb_start = bb_start[id_mask]
    this_bb_stop = bb_stop[id_mask]

    # map all coms to chunks
    chunk_mapping = _centers_to_chunks(blocking, this_ids, this_com,
                                       this_bb_start, this_bb_stop)

    anchor_corrections = {}
    for chunk_id, label_info in chunk_mapping.items():
        corrections = _correct_anchors_for_chunk(ds, blocking, chunk_id, label_info)
        anchor_corrections.update(corrections)
    return anchor_corrections


def correct_anchors(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    morphology_path = config['morphology_path']
    morphology_key = config['morphology_key']

    tmp_folder = config['tmp_folder']
    out_path = os.path.join(tmp_folder, 'corrections_job%i.json' % job_id)

    block_list = config['block_list']
    id_chunks = config['id_chunks']
    number_of_labels = int(vu.file_reader(input_path, 'r')[input_key].attrs['maxId']) + 1
    id_blocking = nt.blocking([0], [number_of_labels], [id_chunks])

    # load the morphology and get the relevant cols
    with vu.file_reader(morphology_path) as f:
        ds = f[morphology_key]
        morphology = ds[:]
    label_ids = morphology[:, 0].astype('uint64')
    assert int(label_ids.max() + 1) == number_of_labels, "%i, %i" % (int(label_ids.max() + 1),
                                                                     number_of_labels)
    com = morphology[:, 2:5]
    bb_start = morphology[:, 5:8].astype('uint64')
    bb_stop = morphology[:, 8:11].astype('uint64')

    with vu.file_reader(input_path, 'r') as f:
        ds = f[input_key]
        blocking = nt.blocking([0, 0, 0], list(ds.shape), list(ds.chunks))

        anchor_corrections = {}
        for block_id in block_list:
            block = id_blocking.getBlock(block_id)
            label_begin = block.begin[0]
            label_end = block.end[0]
            corrections = _correct_anchors_for_label_range(ds, blocking,
                                                           label_ids, com, bb_start, bb_stop,
                                                           label_begin, label_end)
            anchor_corrections.update(corrections)

    fu.log("Job %i: Found %i labels that need anchor corrections" % (job_id, len(anchor_corrections)))
    fu.log("Saving corrections to %s" % out_path)
    with open(out_path, 'w') as f:
        json.dump(anchor_corrections, f)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    correct_anchors(job_id, path)
