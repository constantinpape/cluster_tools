import os
from concurrent import futures
from itertools import product

import numpy as np
import z5py
import nifty
import nifty.distributed as ndist
import skeletor.io
# from skeletor import simplify_skeleton
from .io import write_edge_result


def to_coords(nodes, node_ids):
    coords = nodes[node_ids]
    return tuple(np.array([coord[i] for coord in coords]) for i in range(3))


# TODO which strategy ?
# there are several strategies to from skeleton to edges/costs
# a) introduce edges between all terminals
# b) introduce edges for all junctions
# ...
# subsampling number of edges?
def skeleton_to_edges(seg_id, ds_ws, ds_skel,
                      gt_labels, assignment,
                      rag, ds_out):
    """
    """

    print("Computing skeleton edges for object", seg_id, "/", ds_skel.shape[0])

    # load the skeleton
    nodes, edges = skeletor.io.read_n5(ds_skel, seg_id)
    if nodes is None:
        return

    # get the bounding box of all skeleton nodes and load seg and gt
    mins = np.min(nodes, axis=0)
    maxs = np.max(nodes, axis=0)
    nodes -= mins
    bb = tuple(slice(int(mi), int(ma) + 1)
               for mi, ma in zip(mins, maxs))
    ws = ds_ws[bb]

    # for now, use strategy a)

    # find all terminal nodes
    n_nodes = int(edges.max()) + 1
    g = nifty.graph.undirectedGraph(n_nodes)
    g.insertEdges(edges)
    degrees = np.array([len([adj for adj in g.nodeAdjacency(u)])
                        for u in range(n_nodes)])
    terminals = np.where(degrees == 1)[0]
    # print("terminals")
    # print(terminals)

    # compute all pairs of termina nodes and corresponding coordinates
    terminal_pairs = product(terminals, terminals)
    terminal_pairs = np.array([list(pair) for pair in terminal_pairs if pair[0] < pair[1]])
    # print("terminal pairs")
    # print(terminal_pairs)

    # a single terminal -> no terminal pairs can happen (probably when we have a loop).
    # need to skip it, because will fail in next step
    if terminal_pairs.size == 0:
        return

    coords1 = to_coords(nodes, terminal_pairs[:, 0])
    coords2 = to_coords(nodes, terminal_pairs[:, 1])

    # map terminal nodes to watershed ids (= rag ids)
    lifted_uvs = np.concatenate([ws[coords1][:, None], ws[coords2][:, None]],
                                axis=1).astype('uint64')
    # sort the uv pairs
    lifted_uvs = np.sort(lifted_uvs, axis=1)

    # get rid of edges to watershed ids that are not part of the assignments
    # (this can happen due to issues in the skeletonization)
    this_fragments = np.where(assignment == seg_id)[0]
    assignment_mask = np.isin(lifted_uvs, this_fragments).all(axis=1)
    lifted_uvs = lifted_uvs[assignment_mask]
    if lifted_uvs.size == 0:
        return

    # get rid of self-edges
    mask = lifted_uvs[:, 0] != lifted_uvs[:, 1]
    lifted_uvs = lifted_uvs[mask]
    if lifted_uvs.size == 0:
        return

    # print("lifted uvs")
    # print(lifted_uvs)
    # print(lifted_uvs.shape)
    # make the edges unique
    lifted_uvs = np.unique(lifted_uvs, axis=0)
    # print("lifted uvs")
    # print(lifted_uvs)
    # print(lifted_uvs.shape)

    # filter by rag edges
    if rag is not None:
        lifted_uvs = lifted_uvs[rag.findEdges(lifted_uvs) == -1]
        if lifted_uvs.size == 0:
            return

    # get the ground-truth mapping
    lu, lv = gt_labels[lifted_uvs[:, 0]], gt_labels[lifted_uvs[:, 1]]
    # filter for nodes mapped to ignore label
    gt_mask = np.logical_and(lu != 0, lv != 0)
    lifted_uvs = lifted_uvs[gt_mask]
    if lifted_uvs.size == 0:
        return
    # print("lifted uvs")
    # print(lifted_uvs)

    lu, lv = lu[gt_mask], lv[gt_mask]
    assert len(lu) == len(lv) == len(lifted_uvs)
    assert (lu != 0).all()
    assert (lv != 0).all()

    # check for lifted edges with false merge
    merge_indicators = (lu != lv)
    # return

    # write to custom n5 varlen output
    write_edge_result(ds_out, seg_id, lifted_uvs, merge_indicators)


def edges_from_skeletons(path, ws_key, labels_key,
                         skel_key, assignment_key, out_key,
                         graph_path, graph_key, n_threads):
    f = z5py.File(path)
    ds_ws = f[ws_key]
    ds_skel = f[skel_key]
    n_labels = ds_skel.shape[0]

    ds_labels = f[labels_key]
    ds_labels.n_threads = n_threads
    gt_labels = ds_labels[:]

    ds_assignment = f[assignment_key]
    ds_assignment.n_threads = n_threads
    assignment = ds_assignment[:]

    rag = ndist.Graph(os.path.join(graph_path, graph_key), n_threads)
    ds_out = f.require_dataset(out_key, shape=(n_labels,), chunks=(1,), compression='gzip',
                               dtype='uint64')

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(skeleton_to_edges, seg_id, ds_ws, ds_skel,
                           gt_labels, assignment, rag, ds_out)
                 for seg_id in range(n_labels)]
        [t.result() for t in tasks]

    # for seg_id in range(n_labels):
    #     skeleton_to_edges(seg_id, ds_ws, ds_skel, gt_labels, rag, ds_out)
    # seg_id = 4756
    # skeleton_to_edges(seg_id, ds_ws, ds_skel, gt_labels, rag, ds_out)
