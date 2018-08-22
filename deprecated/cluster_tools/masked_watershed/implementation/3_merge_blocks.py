#! /usr/bin/python

import os
import time
import argparse
import numpy as np
import h5py

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

    # need additional attributes to deterimine the actual overlap
    with h5py.File(path_a, 'r') as f:
        attrs = f['data'].attrs
        # we should maybe sanity check that these agree for block b
        ovlp_dim = attrs['overlap_dimension']
        ovlp_begin = attrs['overlap_begin']
        ovlp_end = attrs['overlap_end']

    # find the ids ON the actual block boundary
    ovlp_len = ovlp_a.shape[ovlp_dim]
    ovlp_dim_begin = ovlp_len // 2 if ovlp_len % 2 == 1 else ovlp_len // 2 - 1
    ovlp_dim_end = ovlp_len // 2 + 1
    boundary = tuple(slice(ovlp_begin[i], ovlp_end[i]) if i != ovlp_dim else
                     slice(ovlp_dim_begin, ovlp_dim_end) for i in range(3))

    # measure all overlaps
    overlaps_ab = ngt.overlap(ovlp_a, ovlp_b)
    overlaps_ba = ngt.overlap(ovlp_b, ovlp_a)
    node_assignment = []

    # find the ids ON the actual block boundary
    segments_a = np.unique(ovlp_a[boundary])
    segments_b = np.unique(ovlp_b[boundary])

    for seg_a in segments_a:

        # skip ignore label
        if seg_a == 0:
            continue
        ovlp_seg_a, counts_seg_a = overlaps_ab.overlapArraysNormalized(seg_a, sorted=True)

        seg_b = ovlp_seg_a[0]
        # skip ignore label
        if seg_b == 0:
            continue

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


def masked_watershed_step3(input_file, tmp_folder, ovlp_threshold=.95):
    t0 = time.time()
    overlap_ids = np.load(input_file)
    offsets = np.load(os.path.join(tmp_folder, 'offsets.npy'))

    # get the node assignments from block overlaps
    results = [merge_blocks(ovlp_ids, tmp_folder, offsets, ovlp_threshold) for ovlp_ids in overlap_ids]

    # concatenate results
    node_assignment = [res for res in results if res is not None]
    # we may have no node assignments for overlaps that are completely covered
    # by the mask
    if node_assignment:
        node_assignment = np.concatenate(node_assignment, axis=0)

    job_id = int(os.path.split(input_file)[1].split('_')[3][:-4])
    np.save(os.path.join(tmp_folder, '3_output_assignments_%i.npy' % job_id), node_assignment)

    print("Success %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()
    masked_watershed_step3(args.input_file, args.tmp_folder)
