import os
import time
from concurrent import futures
import numpy as np

import vigra
import z5py
import nifty


def connected_components_ufd(path, key,
                             out_path, out_key,
                             block_shape, out_chunks,
                             tmp_folder, n_threads):
    # TODO assert that block shape is multiple of chunks
    n5_in = z5py.File(path, use_zarr_format=False)
    ds = n5_in[key]
    shape = ds.shape

    n5_out = z5py.File(out_path, use_zarr_format=False)
    if out_key not in n5_out:
        ds_out = n5_out.create_dataset(out_key, dtype='uint64',
                                       shape=shape, chunks=out_chunks,
                                       compression='gzip')
    else:
        ds_out = n5_out[out_key]
        assert ds_out.shape == shape
        assert ds_out.chunks == out_chunks

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    halo = [1, 1, 1]
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    #
    # First pass:
    # We run connected components for each block with an overlap of 1 in each direction
    # We serialize the block result (without overlap) and serialize all the overlaps
    #
    def cc_block(block_id):
        # get all the relevant blocks
        block = blocking.getBlockWithHalo(block_id, halo)
        inner_block, outer_block, local_block = block.innerBlock, block.outerBlock, block.innerBlockLocal
        outer_shape = outer_block.shape

        # we offset with the coordinate of the leftmost pixel
        offset = sum(e * s for e, s in zip(inner_block.begin, shape))

        # get all bounding boxes
        bb_outer = tuple(slice(b, e) for b, e in zip(outer_block.begin, outer_block.end))
        begin, end = inner_block.begin, inner_block.end
        bb_inner = tuple(slice(b, e) for b, e in zip(begin, end))
        bb_local = tuple(slice(b, e) for b, e in zip(local_block.begin, local_block.end))
        total_shape = outer_block.shape

        # get the subvolume, find connected components and write non-overlapping part to file
        subvolume = ds[bb_outer]
        cc = vigra.analysis.labelVolumeWithBackground(subvolume).astype('uint64')
        cc[cc != 0] += offset
        ds_out[bb_inner] = cc[bb_local]

        # serialize all the overlaps
        overlap_ids = []
        for ii in range(6):
            axis = ii // 2
            to_lower = ii % 2
            neighbor_id = blocking.getNeighborId(block_id, axis=axis, lower=to_lower)

            if neighbor_id != -1:
                overlap_bb = tuple(slice(1 if begin[i] != 0 else 0,
                                         outer_shape[i] - 1 if end[i] != shape[i] else outer_shape[i]) if i != axis else
                                   slice(0, 2) if to_lower else
                                   slice(total_shape[i] - 2, total_shape[i]) for i in range(3))

                overlap = cc[overlap_bb]
                vigra.writeHDF5(overlap, os.path.join(tmp_folder, 'block_%i_%i.h5' % (block_id, neighbor_id)),
                                'data', compression='gzip')

                # we only return the overlap ids, if the block id is smaller than the neighbor id,
                # to keep the pairs unique
                if block_id < neighbor_id:
                    overlap_ids.append((block_id, neighbor_id))
        max_id = int(cc.max())
        return overlap_ids, max_id

    t1 = time.time()
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(cc_block, block_id)
                 for block_id in range(blocking.numberOfBlocks)]
        result = [t.result() for t in tasks]
        # overlap_ids = list(chain(res[0] for res in result))
        overlap_ids = [ids for res in result for ids in res[0]]
        max_id = np.max([res[1] for res in result])
    print("First pass took", time.time() - t1, "s")

    #
    # Second pass:
    # We iterate over all pairs of overlaps and find the node assignments
    #

    def process_overlap(ovlp_ids):
        id_a, id_b = ovlp_ids
        ovlp_a = vigra.readHDF5(os.path.join(tmp_folder, 'block_%i_%i.h5' % (id_a, id_b)), 'data')
        ovlp_b = vigra.readHDF5(os.path.join(tmp_folder, 'block_%i_%i.h5' % (id_b, id_a)), 'data')
        if ovlp_a.shape != ovlp_b.shape:
            print(id_a, id_b)
            assert ovlp_a.shape == ovlp_b.shape, "%s, %s" % (str(ovlp_a.shape), str(ovlp_b.shape))
        # match the non-zero ids
        labeled = ovlp_a != 0
        ids_a, ids_b = ovlp_a[labeled], ovlp_b[labeled]
        node_assignment = np.concatenate([ids_a[None], ids_b[None]], axis=0).transpose()
        if node_assignment.size:
            node_assignment = np.unique(node_assignment, axis=0)
            return node_assignment

    t2 = time.time()
    # with futures.ThreadPoolExecutor(n_threads) as tp:
    #     tasks = [tp.submit(process_overlap, ovlp_ids) for ovlp_ids in overlap_ids]
    #     result = [t.result() for t in tasks]
    #     result = [res for res in result if res is not None]
    #     node_assignment = np.concatenate(result, axis=0)
    result = [process_overlap(ovlp_ids) for ovlp_ids in overlap_ids]
    node_assignment = np.concatenate(result, axis=0)
    print("Second pass took", time.time() - t2, "s")

    #
    # Get unique labeling with union find
    #
    t3 = time.time()
    ufd = nifty.ufd.ufd(max_id + 1)
    ufd.merge(node_assignment)
    # TODO do we need extra treatment for zeros ?
    node_labeling = ufd.elementLabeling()
    # TODO make labeling consecutive
    print("Ufd took", time.time() - t3, "s")

    #
    # Third pass:
    # We assign all the nodes their final ids for each block
    #
    def assign_node_ids(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
        subvol = ds_out[bb]
        # map node-ids in block
        # TODO this can be done faster !!!!!
        # maybe we should do this in nifty tools
        sub_ids = np.unique(subvol)
        sub_ids = sub_ids[sub_ids != 0]
        for sub_id in sub_ids:
            subvol[subvol == sub_id] = node_labeling[sub_id]
        ds_out[bb] = subvol

    t3 = time.time()
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(assign_node_ids, block_id)
                 for block_id in range(blocking.numberOfBlocks)]
    print("Second pass took", time.time() - t3, "s")
