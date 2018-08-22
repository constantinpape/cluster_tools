import os
# import time
from concurrent import futures
import numpy as np

import vigra
import z5py
import nifty
import nifty.ground_truth as ngt

import h5py
# from cremi_tools.viewer.volumina import view


def watershed(aff_path_xy, key_xy, aff_path_z, key_z, out_path, key_out,
              out_chunks, out_blocks, tmp_folder, halo=[5, 50, 50],
              threshold_cc=.95, threshold_dt=.8, sigma_seeds=1.):
    assert os.path.exists(aff_path_xy)
    assert os.path.exists(aff_path_z)
    assert all(block % chunk == 0 for chunk, block in zip(out_chunks, out_blocks))
    ds_xy = z5py.File(aff_path_xy)[key_xy]
    ds_z = z5py.File(aff_path_z)[key_z]

    shape = ds_xy.shape
    assert ds_z.shape == shape

    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    f_out = z5py.File(out_path, use_zarr_format=False)
    if key_out in f_out:
        ds_out = f_out[key_out]
        assert ds_out.chunks == out_chunks, "%s, %s" % (str(ds_out.chunks), str(out_chunks))
        assert ds_out.shape == shape
    else:
        ds_out = f_out.create_dataset(key_out, shape=shape, chunks=out_chunks, dtype='uint64',
                                      compression='gzip')

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(out_blocks))

    def ws_block(block_id):
        print("Start", block_id, "/", blocking.numberOfBlocks)
        block = blocking.getBlockWithHalo(block_id, halo)

        # get all blockings
        outer_block, inner_block, local_block = block.outerBlock, block.innerBlock, block.innerBlockLocal
        inner_bb = tuple(slice(b, e) for b, e in zip(inner_block.begin, inner_block.end))
        outer_bb = tuple(slice(b, e) for b, e in zip(outer_block.begin, outer_block.end))
        local_bb = tuple(slice(b, e) for b, e in zip(local_block.begin, local_block.end))

        # load the data and average it for z affinities
        affs_xy = ds_xy[outer_bb]
        affs_z = ds_z[outer_bb]
        # affs_z += affs_xy
        # affs_z /= 2.

        # generate seeds from thresholded connected components
        thresholded = affs_z > threshold_cc
        # TODO to be more conservative, we could filter again by 2d connected components
        # (i. e. things that are not connected in 2d get chopped off)
        seeds = vigra.analysis.labelVolumeWithBackground(thresholded.view('uint8'))
        seed_offset = seeds.max() + 1

        # generate seeds from distance transform in 2d
        thresholded_dt = (affs_xy < threshold_dt).astype('uint32')
        seeds_dt = np.zeros_like(thresholded_dt, dtype='uint32')
        offset_z = 0
        for z in range(seeds_dt.shape[0]):
            dt = vigra.filters.distanceTransform(thresholded_dt[z])
            if sigma_seeds > 0.:
                dt = vigra.filters.gaussianSmoothing(dt, sigma_seeds)
            seeds_z = vigra.analysis.localMaxima(dt, allowPlateaus=True, allowAtBorder=True, marker=np.nan)
            seeds_z = vigra.analysis.labelImageWithBackground(np.isnan(seeds_z).view('uint8'))
            seeds_z[seeds_z != 0] += offset_z
            offset_z = seeds_z.max() + 1
            seeds_dt[z] = seeds_z

        # merge seeds
        seeds_dt[seeds_dt != 0] += seed_offset
        no_seed_mask = seeds == 0
        seeds[no_seed_mask] = seeds_dt[no_seed_mask]

        # run watersheds in 2d
        ws = np.zeros_like(seeds, dtype='uint32')
        affs_xy = 1. - affs_xy
        for z in range(ws.shape[0]):
            ws[z] = vigra.analysis.watershedsNew(affs_xy[z], seeds=seeds[z])[0]

        # filter tiny components
        size_filter = 25
        ids, sizes = np.unique(ws, return_counts=True)
        mask = np.ma.masked_array(ws, np.in1d(ws, ids[sizes < size_filter])).mask
        ws[mask] = 0
        ws, _ = vigra.analysis.watershedsNew(affs_xy, seeds=ws)
        ws, max_id, _ = vigra.analysis.relabelConsecutive(ws)

        # save the inner block
        ds_out[inner_bb] = ws[local_bb].astype('uint64')

        # serialize the overlaps
        overlap_ids = []
        for ii in range(6):
            axis = ii // 2
            to_lower = ii % 2
            neighbor_id = blocking.getNeighborId(block_id, axis=axis, lower=to_lower)

            if neighbor_id != -1:
                overlap_bb = tuple(slice(None) if i != axis else
                                   slice(0, 2*halo[i]) if to_lower else
                                   slice(inner_block.end[i] - halo[i] - outer_block.begin[i],
                                         outer_block.end[i] - outer_block.begin[i]) for i in range(3))

                overlap_coords = tuple((outer_block.begin[i], outer_block.end[i]) if i != axis else
                                       (outer_block.begin[i], inner_block.begin[i] + halo[i]) if to_lower else
                                       (inner_block.end[i] - halo[i], outer_block.end[i]) for i in range(3))

                overlap = ws[overlap_bb]

                ovlp_path = os.path.join(tmp_folder, 'block_%i_%i.h5' % (block_id, neighbor_id))
                vigra.writeHDF5(overlap, ovlp_path, 'data', compression='gzip')
                with h5py.File(ovlp_path) as f:
                    f['data'].attrs['coords'] = overlap_coords

                # we only return the overlap ids, if the block id is smaller than the neighbor id,
                # to keep the pairs unique
                if block_id < neighbor_id:
                    overlap_ids.append((block_id, neighbor_id))

        return max_id, overlap_ids

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(ws_block, block_id)
                 for block_id in range(blocking.numberOfBlocks)]
        results = [t.result() for t in tasks]
    offsets = np.array([res[0] for res in results], dtype='uint64')
    overlap_ids = [ids for res in results for ids in res[1]]

    last_max_id = offsets[0]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    max_id = offsets[-1] + last_max_id

    ovlp_threshold = .9

    def merge_blocks(ovlp_ids):
        id_a, id_b = ovlp_ids
        path_a = os.path.join(tmp_folder, 'block_%i_%i.h5' % (id_a, id_b))
        path_b = os.path.join(tmp_folder, 'block_%i_%i.h5' % (id_b, id_a))
        ovlp_a = vigra.readHDF5(path_a, 'data')
        ovlp_b = vigra.readHDF5(path_b, 'data')
        offset_a, offset_b = offsets[id_a], offsets[id_b]
        ovlp_a += offset_a
        ovlp_b += offset_b

        with h5py.File(path_a) as f:
            coords_a = f['data'].attrs['coords']
        with h5py.File(path_b) as f:
            coords_b = f['data'].attrs['coords']

        if ovlp_a.shape != ovlp_b.shape:
            print(coords_a)
            print(coords_b)
            assert ovlp_a.shape == ovlp_b.shape, "%s, %s" % (str(ovlp_a.shape), str(ovlp_b.shape))

        # bb_a = tuple(slice(c_a[0], c_a[1]) for c_a in coords_a)
        # bb_b = tuple(slice(c_b[0], c_b[1]) for c_b in coords_b)
        # affs_a = ds_xy[bb_a]
        # affs_b = ds_xy[bb_b]
        # view([affs_a, ovlp_a, affs_b, ovlp_b], ['affs_a', 'seg_b', 'affs_b', 'seg_b'])
        # # quit()

        # measure all overlaps
        segments_a = np.unique(ovlp_a)
        overlaps_ab = ngt.overlap(ovlp_a, ovlp_b)
        overlaps_ba = ngt.overlap(ovlp_b, ovlp_a)
        node_assignment = []
        for seg_a in segments_a:
            ovlp_seg_a, counts_seg_a = overlaps_ab.overlapArraysNormalized(seg_a, sorted=True)
            seg_b = ovlp_seg_a[0]
            ovlp_seg_b, counts_seg_b = overlaps_ba.overlapArraysNormalized(seg_b, sorted=True)
            if ovlp_seg_b[0] != seg_a:
                continue

            ovlp_measure = (counts_seg_a[0] + counts_seg_b[0]) / 2.
            if ovlp_measure > ovlp_threshold:
                node_assignment.append([seg_a, seg_b])

        if node_assignment:
            return np.array(node_assignment, dtype='uint64')
        else:
            return None

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(merge_blocks, ovlp_ids) for ovlp_ids in overlap_ids]
        results = [t.result() for t in tasks]
    # result = [merge_blocks(ovlp_ids) for ovlp_ids in overlap_ids]
    node_assignment = np.concatenate([res for res in results if res is not None], axis=0)

    ufd = nifty.ufd.ufd(max_id + 1)
    ufd.merge(node_assignment)
    node_labeling = ufd.elementLabeling()

    def assign_node_ids(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
        subvol = ds_out[bb]
        offset = offsets[block_id]
        subvol += offset
        ds_out[bb] = nifty.tools.take(node_labeling, subvol)

    with futures.ThreadPoolExecutor(8) as tp:
        tasks = [tp.submit(assign_node_ids, block_id) for block_id in range(blocking.numberOfBlocks)]
