#! /usr/bin/python

import os
import time
import argparse
import numpy as np

import vigra
import nifty.ground_truth as ngt


def merge_blocks(ovlp_ids, tmp_folder, offsets, ovlp_threshold):
    id_a, id_b = ovlp_ids
    path_a = os.path.join(tmp_folder, 'block_%i_%i.h5' % (id_a, id_b))
    path_b = os.path.join(tmp_folder, 'block_%i_%i.h5' % (id_b, id_a))
    ovlp_a = vigra.readHDF5(path_a, 'data')
    ovlp_b = vigra.readHDF5(path_b, 'data')
    offset_a, offset_b = offsets[id_a], offsets[id_b]

    assert ovlp_a.shape == ovlp_b.shape, "%s, %s" % (str(ovlp_a.shape), str(ovlp_b.shape))

    segments_a = np.unique(ovlp_a)
    overlaps_ab = ngt.overlap(ovlp_a, ovlp_b)
    overlaps_ba = ngt.overlap(ovlp_b, ovlp_a)

    # find the ids ON the actual block boundary
    boundary_mask = ''  # TODO
    segments_a = np.unique(ovlp_a[boundary_mask])
    segments_b = np.unique(ovlp_b[boundary_mask])

    node_assignment = []
    for seg_a in segments_a:
        ovlp_seg_a, counts_seg_a = overlaps_ab.overlapArraysNormalized(seg_a, sorted=True)
        seg_b = ovlp_seg_a[0]
        ovlp_seg_b, counts_seg_b = overlaps_ba.overlapArraysNormalized(seg_b, sorted=True)
        if ovlp_seg_b[0] != seg_a or seg_b not in segments_b:
            continue

        ovlp_measure = (counts_seg_a[0] + counts_seg_b[0]) / 2.
        if ovlp_measure > ovlp_threshold:
            node_assignment.append([seg_a + offset_a, seg_b + offset_b])

    if node_assignment:
        return np.array(node_assignment, dtype='uint64')
    else:
        return None


def watershed_step3(input_file, tmp_folder, ovlp_threshold=.9):
    t0 = time.time()
    overlap_ids = np.load(input_file)
    offsets = np.load(os.path.join(tmp_folder, 'offsets.npy'))
    results = [merge_blocks(ovlp_ids, tmp_folder, offsets, ovlp_threshold) for ovlp_ids in overlap_ids]
    node_assignment = np.concatenate([res for res in results if res is not None], axis=0)
    job_id = int(os.path.split(input_file)[1].split('_')[3][:-4])
    np.save(os.path.join(tmp_folder, '3_output_assignments_%i.npy' % job_id), node_assignment)

    print("Success %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()
    watershed_step3(args.input_file, args.tmp_folder)
