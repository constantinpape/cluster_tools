#! /usr/bin/python

import time
import os
import argparse

import z5py
import nifty
import cremi_tools.segmentation as cseg

# TODO support more agglomerators
AGGLOMERATORS = {"multicut_kl": cseg.Multicut("kernighan-lin")}


# TODO multithreaded
def multicut_step2(out_path, node_labeling_key,
                   n_scales, tmp_folder, agglomerator_key):
    t0 = time.time()
    last_scale = n_scales
    agglomerator = AGGLOMERATORS[agglomerator_key]

    f_graph = z5py.File(os.path.join(tmp_folder, 'merged_graph.n5/s%i' % last_scale))
    n_nodes = f_graph.attrs['numberOfNodes']
    uv_ids = f_graph['edges'][:]
    initial_node_labeling = f_graph['nodeLabeling'][:]
    n_edges = len(uv_ids)

    # get the costs
    costs = f_graph['costs'][:]
    assert len(costs) == n_edges, "%i, %i" (len(costs), n_edges)

    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    node_labeling = agglomerator(graph, costs)

    # get the labeling of initial nodes
    if initial_node_labeling is None:
        new_initial_node_labeling = node_labeling
    else:
        # should this ever become a bottleneck, we can parallelize this in nifty
        # but for now this would really be premature optimization
        new_initial_node_labeling = node_labeling[initial_node_labeling]

    f_out = z5py.File(out_path, use_zarr_format=False)
    node_shape = (len(new_initial_node_labeling), )
    chunks = (min(len(new_initial_node_labeling), 524288), )
    ds_nodes = f_out.create_dataset(node_labeling_key, dtype='uint64', shape=node_shape,
                                    chunks=chunks)
    ds_nodes[:] = new_initial_node_labeling

    print("Success")
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    parser.add_argument("node_labeling_key", type=str)
    parser.add_argument("n_scales", type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--agglomerator_key", type=str)
    # parser.add_argument("--n_threads", type=int)
    args = parser.parse_args()

    multicut_step2(args.out_path,
                   args.node_labeling_key,
                   args.n_scales,
                   args.tmp_folder, args.agglomerator_key)
