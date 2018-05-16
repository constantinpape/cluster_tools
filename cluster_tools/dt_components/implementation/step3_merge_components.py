#! /usr/bin/python

import argparse
import json
import os

import numpy as np
import z5py
import nifty


def merge_block(ds, block_id, blocking, offsets, empty_blocks):
    # get current block
    block = blocking.getBlockWithHalo(block_id)
    off = offsets[block_id]

    # iterate over the neighbors, find adjacent component
    # and merge them
    assignments = []
    to_lower = False
    for dim in range(3):

        # find the block id of the overlapping neighbor
        ngb_id = blocking.getNeighborId(block_id, dim, to_lower)
        if ngb_id == -1:
            continue
        # don't stitch with empty blocks
        if ngb_id in empty_blocks:
            continue

        # find the bb for the adjacent faces of both blocks
        adjacent_bb = tuple(slice(block.begin[i], block.end[i]) if i != dim else
                            slice(block.end[i] - 1, block.end[i] + 1)
                            for i in range(3))
        # load the adjacent faces
        adjacent_faces = ds[adjacent_bb]

        # get the face of this block and the ngb block
        bb = tuple(slice(None) if i != dim else slice(0, 1) for i in range(3))
        bb_ngb = tuple(slice(None) if i != dim else slice(1, 2) for i in range(3))
        face, face_ngb = adjacent_faces[bb], adjacent_faces[bb_ngb]

        # add the offsets
        labeled = face != 0
        face[labeled] += off
        off_ngb = offsets[ngb_id]
        face_ngb[face_ngb != 0] += off_ngb

        # find the assignments via touching (non-zero) component ids
        ids_a, ids_b = face[labeled], face_ngb[labeled]
        assignment = np.concatenate([ids_a[None], ids_b[None]], axis=0).transpose()
        if assignment.size:
            assignment = np.unique(assignment, axis=0)
            # filter zero assignments
            valid_assignment = (assignment != 0).all(axis=1)
            assignments.append(assignment[valid_assignment])

    if assignments:
        return np.concateate(assignments, axis=0)
    else:
        return None


def step3_merge_components(path, out_key, cache_folder, job_id):

    # we re-use the config from step 1, but only use the block-ids
    input_file = os.path.join(cache_folder, '1_config_%i.json' % job_id)
    with open(input_file) as f:
        input_config = json.load(f)
        block_shape = input_config['block_shape']
        block_ids = list(input_config['block_config'].keys())

    # load the block offsets and empty blocks
    with open(os.path.join(cache_folder, 'block_offsets.json')) as f:
        offset_config = json.load(f)
    empty_blocks = offset_config['empty_blocks']
    offsets = offset_config['offsets']

    ds = z5py.File(path)[out_key]
    shape = ds.shape
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape), empty_blocks)

    results = [merge_block(ds, block_id, blocking, offsets)
               for block_id in block_ids
               if block_id not in empty_blocks]

    results = [res for res in results if res is not None]
    if results:
        results = np.concatenate(results, axis=0)
    out_path = os.path.join(cache_folder, '3_results_%i.npy' % job_id)
    np.save(out_path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('job_id', type=int)
    args = parser.parse_args()
    step3_merge_components(args.path, args.out_key, args.cache_folder, args.job_id)
