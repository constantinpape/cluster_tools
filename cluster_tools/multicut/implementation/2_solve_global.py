#! /usr/bin/python

import time
import os
import argparse
import numpy as np

import z5py
import nifty
import cremi_tools.segmentation as cseg
import nifty.distributed as ndist

# TODO support more agglomerators
AGGLOMERATORS = {"multicut_kl": cseg.Multicut("kernighan-lin")}


def multicut_step2(out_path, nodes_key, n_scales, tmp_folder, agglomerator_key):
    last_scale = n_scales - 1

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
    node_shape = (len(new_initial_node_labelingn))
    ds_nodes = f_out.create_dataset('nodeLabeling', dtype='uint64', shape=node_shape, chunks=node_shape)
    f_out[:] = new_initial_node_labeling
