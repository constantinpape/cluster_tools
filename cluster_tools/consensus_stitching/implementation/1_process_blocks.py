#! /usr/bin/python

import os
import argparse
import json

# import vigra
import numpy as np
import nifty
import nifty.graph.rag as nrag
import z5py
import cremi_tools.segmentation as cseg


def segment_block(ds_ws, ds_affs, blocking, block_id, offsets):

    # load the segmentation
    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    ws = ds_ws[bb]

    # if this block only contains a single segment id (usually 0 = ignore label) continue
    ws_ids = np.unique(ws)
    if len(ws_ids) == 1:
        return None
    max_id = ws_ids[-1]

    # TODO should we do this ?
    # map to a consecutive segmentation to speed up graph computations
    # ws, max_id, mapping = vigra.analysis.relabelConsecutive(ws, keep_zeros=True, start_label=1)

    # load the affinities
    n_channels = len(offsets)
    bb_affs = (slice(0, n_channels),) + bb
    affs = ds_affs[bb_affs]
    # convert affinities to float and invert them
    # to get boundary probabilities
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = 1. - affs

    # compute the region adjacency graph
    n_labels = int(max_id) + 1
    rag = nrag.gridRag(ws,
                       numberOfLabels=n_labels,
                       numberOfThreads=1)
    uv_ids = rag.uvIds()

    # compute the features and get edge probabilities (from mean affinities)
    # and edge sizes
    features = nrag.accumulateAffinityStandartFeatures(rag, affs, offsets,
                                                       numberOfThreads=1)
    probs = features[:, 0]
    sizes = features[:, -1].astype('uint64')

    # compute multicut
    mc = cseg.Multicut('kernighan-lin', weight_edges=False)
    # transform probabilities to costs
    costs = mc.probabilities_to_costs(probs)
    # set edges connecting to 0 (= ignore label) to repulsive
    ignore_edges = (uv_ids == 0).any(axis=1)
    costs[ignore_edges] = -100
    # solve the mc problem
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uv_ids)
    node_labels = mc(graph, costs)

    # get indicators for merge !
    # and return uv-ids, edge indicators and edge sizes
    edge_indicator = (node_labels[uv_ids[:, 0]] == node_labels[uv_ids[:, 1]]).astype('uint8')

    return uv_ids, edge_indicator, sizes


def step1(path, ws_key, aff_key, cache_folder, job_id, prefix):

    # load the blocks to be processed and the configuration from the input config file
    with open(os.path.join(cache_folder, '%s_config_%i.json' % (prefix, job_id))) as f:
        input_config = json.load(f)
    block_list = input_config['blocks']
    config = input_config['config']
    block_shape, block_shift = config['block_shape'], config['block_shift']
    offsets = config['offsets']
    weight_edges = config['weight_edges']

    # open all n5 datasets
    ds_ws = z5py.File(path)[ws_key]
    ds_affs = z5py.File(path)[aff_key]
    shape = ds_ws.shape

    blocking = nifty.tools.blocking([0, 0, 0], list(shape), block_shape,
                                    blockShift=block_shift)

    results = [segment_block(ds_ws, ds_affs,
                             blocking, block_id, offsets)
               for block_id in block_list]
    results = [res for res in results if res is not None]

    # all blocks could be empty
    if not results:
        merged_uvs, merge_votes = [], []

    else:
        uv_ids = np.concatenate([res[0] for res in results], axis=0)
        indicators = np.concatenate([res[1] for res in results], axis=0)
        sizes = np.concatenate([res[2] for res in results], axis=0)

        # compute nominator and denominator of merge votes
        merged_uvs, merge_votes = nifty.tools.computeMergeVotes(uv_ids, indicators, sizes,
                                                                weightEdges=weight_edges)

    # serialize the results
    save_path1 = os.path.join(cache_folder, '%s_votes_%i.npy' % (prefix, job_id))
    np.save(save_path1, merge_votes)
    save_path2 = os.path.join(cache_folder, '%s_uvs_%i.npy' % (prefix, job_id))
    np.save(save_path2, merged_uvs)

    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('ws_key', type=str)
    parser.add_argument('aff_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('job_id', type=int)
    parser.add_argument('prefix', type=str)

    args = parser.parse_args()
    step1(args.path,
          args.ws_key,
          args.aff_key,
          args.cache_folder,
          args.job_id,
          args.prefix)
