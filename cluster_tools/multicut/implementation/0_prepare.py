#! /usr/bin/python
import os
import argparse

import z5py
import cremi_tools.segmentation as cseg


def make_costs(feature_path, feature_key):
    pass


def prepare(labels_path, labels_key,
            graph_path, graph_key,
            feature_path,
            out_path, out_key,
            initial_block_shape, n_scales,
            weight_edges=True):

    assert os.path.exists(feature_path), feature_path

    # graph = ndist.loadAsUndirectedGraph(os.path.join(graph_path, graph_key))
    # load number of nodes and uv-ids
    f_graph = z5py.File(graph_path)[graph_key]
    shape = f_graph.attrs['shape']
    n_nodes = f_graph.attrs['numberOfNodes']
    uv_ids = f_graph['edges'][:]

    # get the multicut edge costs from mean affinities
    feature_ds = z5py.File(feature_path)['features']
    costs = 1. - feature_ds[:, 0:1].squeeze()
    if weight_edges:
        edge_sizes = feature_ds[:, 9:].squeeze()
    else:
        edge_sizes = None

    # find ignore edges
    ignore_edges = (uv_ids == 0).any(axis=1)

    # set edge sizes of ignore edges to 1 (we don't want them to influence the weighting)
    edge_sizes[ignore_edges] = 1
    costs = cseg.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)

    # set weights of ignore edges to be maximally repulsive
    costs[ignore_edges] = 5 * costs.min()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("graph_key", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--n_jobs1", type=int)
    parser.add_argument("--n_jobs2", type=int)
    parser.add_argument("--tmp_folder", type=str)
    args = parser.parse_args()

    prepare(args.graph_path, args.graph_key,
            args.out_path, args.out_key,
            list(args.block_shape), args.n_jobs1,
            args.n_jobs2, args.tmp_folder)
