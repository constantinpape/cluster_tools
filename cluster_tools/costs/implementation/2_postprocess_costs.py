#! /usr/bin/python

import argparse
import z5py
import cremi_tools.segmentation as cseg


# TODO multi-threaded implementation
def make_costs(input_path, input_key,
               features_path, features_key,
               ds_graph, costs_for_multicut,
               beta, weighting_exponent,
               weight_edges, invert_inputs):

    # find the ignore edges
    uv_ids = ds_graph['edges'][:]
    ignore_edges = (uv_ids == 0).any(axis=1)

    # get the multicut edge costs from mean affinities
    input_ds = z5py.File(input_path)[input_key]
    # we might have 1d or 2d inputs, depending on input from features or random forest
    slice_ = slice(None) if input_ds.ndim == 1 else tuple(slice(None), slice(0, 1))

    if invert_inputs:
        costs = 1. - input_ds[slice_].squeeze()
    else:
        costs = input_ds[slice_].squeeze()

    if costs_for_multicut:

        if weight_edges:
            # TODO the edge sizes might not be hardcoded to this feature
            # id in the future
            feature_ds = z5py.File(features_path, features_key)
            edge_sizes = feature_ds[:, 9:10].squeeze()
            # set edge sizes of ignore edges to 1 (we don't want them to influence the weighting)
            edge_sizes[ignore_edges] = 1
        else:
            edge_sizes = None

        costs = cseg.transform_probabilities_to_costs(costs, beta=beta,
                                                      edge_sizes=edge_sizes,
                                                      weighting_exponent=weighting_exponent)
    # set weights of ignore edges to be maximally repulsive
    if costs_for_multicut:
        costs[ignore_edges] = 5 * costs.min()
    else:
        costs[ignore_edges] = 1.
    return costs.astype('float32')


# TODO expose default arguments
def costs_step2(input_path, input_key,
                features_path, features_key,
                graph_path, graph_key,
                out_path, out_key,
                invert_inputs,
                costs_for_multicut=True,
                beta=0.5,
                weight_edges=False,
                weighting_exponent=1.):

    ds_graph = z5py.File(graph_path)[graph_key]

    costs = make_costs(input_path, input_key,
                       features_path, features_key,
                       ds_graph,
                       costs_for_multicut,
                       beta, weighting_exponent,
                       weight_edges)
    ds = z5py.File(out_path)[out_key]
    ds[:] = costs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('input_key', type=str)
    parser.add_argument('features_path', type=str)
    parser.add_argument('features_key', type=str)
    parser.add_argument('graph_path', type=str)
    parser.add_argument('graph_key', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('out_key', type=str)

    parser.add_argument('--invert_inputs', type=int)

    args = parser.parse_args()
    costs_step2(args.input_path, args.input_key, args.features_path, args.features_key,
                args.graph_path, args.graph_key, args.out_path, args.out_key,
                args.ingert_inputs)
