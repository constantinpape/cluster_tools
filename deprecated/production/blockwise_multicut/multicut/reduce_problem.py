#! /usr/bin/python

import time
import os
import argparse
import subprocess
import json
import numpy as np
from concurrent import futures

import z5py
import nifty
import nifty.distributed as ndist
import luigi
from vigra.analysis import relabelConsecutive


class ReduceProblemTask(luigi.Task):
    """
    Reduce problem
    """

    graph_path = luigi.Parameter()
    costs_path = luigi.Parameter()
    scale = luigi.IntParameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def run(self):
        from production import util

        # copy the script to the temp folder and replace the shebang
        script_path = os.path.join(self.tmp_folder, 'reduce_problem.py')
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'reduce_problem.py'),
                              script_path)

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            n_threads = config['n_threads']
            roi = config.get('roi', None)

        # prepare the job config
        job_config = {'block_shape': block_shape,
                      'n_threads': n_threads,
                      'roi': roi}
        config_path = os.path.join(self.tmp_folder,
                                   'reduce_problem_config_s%i.json' % self.scale)
        with open(config_path, 'w') as f:
            json.dump(job_config, f)

        command = '%s %s %s %i %s %s' % (script_path, self.graph_path,
                                         self.costs_path, self.scale,
                                         config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder,
                                'logs', 'log_reduce_problem_s%i' % self.scale)
        err_file = os.path.join(self.tmp_folder,
                                'error_logs', 'err_reduce_problem_s%i.err' % self.scale)
        bsub_command = ('bsub -n %i -J reduce_problem ' % n_threads +
                        '-We %i -o %s -e %s \'%s\'' % (self.time_estimate,
                                                       log_file, err_file, command))
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)
            util.wait_for_jobs('papec')

        ds_graph = z5py.File(self.graph_path)['graph']
        nodes = ds_graph.attrs['numberOfNodes']
        edges = ds_graph.attrs['numberOfEdges']
        # check the output
        try:
            with open(self.output().path) as f:
                res = json.load(f)
                t = res['t']
                new_nodes = res['new_nodes']
                new_edges = res['new_edges']
            print("Reduce problem finished in", t, "s")
            print("Reduced number of nodes from", nodes, "to", new_nodes)
            print("Reduced number of edges from", edges, "to", new_edges)
            success = True
        except Exception:
            success = False

        # write output file if we succeed, otherwise write partial
        # success to different file and raise exception
        if not success:
            raise RuntimeError("ReduceProblemTask failed")

    def output(self):
        res_file = os.path.join(self.tmp_folder, 'reduce_problem_s%i.log' % self.scale)
        return luigi.LocalTarget(res_file)


def merge_nodes(tmp_folder, scale, block_list, n_nodes, uv_ids,
                initial_node_labeling, n_threads):
    n_edges = len(uv_ids)

    # load the cut-edge ids from the prev. blocks and make merge edge ids
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(np.load, os.path.join(tmp_folder,
                                                 'subproblem_s%i_%i.npy' % (scale, block_id)))
                 for block_id in block_list]
        res = [t.result() for t in tasks]
        cut_edge_ids = np.concatenate([re for re in res if re.size])
    cut_edge_ids = np.unique(cut_edge_ids).astype('uint64')

    # print("Number of cut edges:", len(cut_edge_ids))
    # print("                   /", n_edges)
    assert len(cut_edge_ids) < n_edges, "%i = %i, does not reduce problem" % (len(cut_edge_ids), n_edges)

    merge_edges = np.ones(n_edges, dtype='bool')
    merge_edges[cut_edge_ids] = False

    # TODO make sure that zero stayes mapped to zero
    # additionally, we make sure that all edges are cut
    ignore_edges = (uv_ids == 0).any(axis=1)
    merge_edges[ignore_edges] = False

    # merge node pairs with ufd
    # FIXME if we process roi's, the number of nodes is underestimated gravely
    n_nodes = int(uv_ids.max()) + 1

    # TODO try relabeling consecutively here to increase processing speed
    # the issue is that we have to do awkward things to get out the complete node
    # labeling again ...
    # uv_ids, n_nodes, mapping = vigra.analysis.relabelConsecutive(uv_ids)
    # inv_mappin = {val: key for key, val in mapping.items()}

    ufd = nifty.ufd.ufd(n_nodes)
    merge_pairs = uv_ids[merge_edges]
    ufd.merge(merge_pairs)

    # get the node results and label them consecutively
    node_labeling = ufd.elementLabeling()
    node_labeling, max_new_id, _ = relabelConsecutive(node_labeling, keep_zeros=True, start_label=1)
    n_new_nodes = max_new_id + 1

    # TODO this is not all we need to do
    # full_node_labeling = np.arange(n_nodes_total, dtype='uint64')
    # node_ids = list(mapping.keys())
    # full_node_labeling[node_ids] = node_labeling

    # get the labeling of initial nodes
    if initial_node_labeling is None:
        new_initial_node_labeling = node_labeling
    else:
        # should this ever become a bottleneck, we can parallelize this in nifty
        # but for now this would really be premature optimization
        new_initial_node_labeling = node_labeling[initial_node_labeling]

    return n_new_nodes, node_labeling, new_initial_node_labeling


def get_new_edges(uv_ids, node_labeling, costs, cost_accumulation, n_threads):
    edge_mapping = nifty.tools.EdgeMapping(uv_ids, node_labeling,
                                           numberOfThreads=n_threads)
    new_uv_ids = edge_mapping.newUvIds()
    edge_labeling = edge_mapping.edgeMapping()
    new_costs = edge_mapping.mapEdgeValues(costs, cost_accumulation, numberOfThreads=n_threads)
    assert len(new_uv_ids) == len(new_costs)
    assert len(edge_labeling) == len(uv_ids)
    return new_uv_ids, edge_labeling, new_costs


def serialize_new_problem(graph_path, n_new_nodes, new_uv_ids,
                          node_labeling, edge_labeling,
                          new_costs, new_initial_node_labeling,
                          shape, scale, initial_block_shape,
                          tmp_folder, n_threads, roi):

    next_scale = scale + 1
    merged_graph_path = os.path.join(tmp_folder, 'merged_graph.n5')
    f_graph = z5py.File(merged_graph_path, use_zarr_format=False)
    # if 's%i' % next_scale not in f_graph:
    #     g_out = f_graph.create_group('s%i' % next_scale)
    # else:
    #     g_out = f_graph['s%i' % next_scale]
    # if 'sub_graphs' not in g_out:
    #     g_out.create_group('sub_graphs')
    g_out = f_graph.require_group('s%i' % next_scale)
    g_out.require_group('sub_graphs')

    # TODO this should be handled by symlinks
    if scale == 0:
        block_in_prefix = os.path.join(graph_path, 'sub_graphs',
                                       's%i' % scale, 'block_')
    else:
        block_in_prefix = os.path.join(tmp_folder, 'merged_graph.n5',
                                       's%i' % scale, 'sub_graphs', 'block_')

    block_out_prefix = os.path.join(tmp_folder, 'merged_graph.n5',
                                    's%i' % next_scale, 'sub_graphs', 'block_')

    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    new_factor = 2**(scale + 1)
    new_block_shape = [new_factor * bs for bs in initial_block_shape]

    shape = z5py.File(graph_path).attrs['shape']
    blocking = nifty.tools.blocking([0, 0, 0], shape, new_block_shape)
    if roi is None:
        new_block_ids = list(range(blocking.numberOfBlocks))
    else:
        new_block_ids = blocking.getBlockIdsOverlappingBoundingBox(roi[0], roi[1],
                                                                   [0, 0, 0]).tolist()

    print("Serializing new graph")
    ndist.serializeMergedGraph(block_in_prefix, shape,
                               block_shape, new_block_shape,
                               new_block_ids, n_new_nodes,
                               node_labeling, edge_labeling,
                               block_out_prefix, n_threads)

    # serialize the full graph for the next scale level
    n_new_edges = len(new_uv_ids)
    g_out.attrs['numberOfNodes'] = n_new_nodes
    g_out.attrs['numberOfEdges'] = n_new_edges

    shape_edges = (n_new_edges, 2)
    ds_edges = g_out.require_dataset('edges', dtype='uint64',
                                     shape=shape_edges, chunks=shape_edges)
    ds_edges.n_threads = n_threads
    ds_edges[:] = new_uv_ids

    nodes = np.unique(new_uv_ids)
    shape_nodes = (len(nodes),)
    ds_nodes = g_out.require_dataset('nodes', dtype='uint64',
                                     shape=shape_nodes, chunks=shape_nodes)
    ds_nodes.n_threads = n_threads
    ds_nodes[:] = nodes

    # serialize the node labeling
    shape_node_labeling = (len(new_initial_node_labeling),)
    ds_node_labeling = g_out.create_dataset('nodeLabeling', dtype='uint64',
                                            shape=shape_node_labeling,
                                            chunks=shape_node_labeling)
    ds_node_labeling[:] = new_initial_node_labeling

    # serialize the new costs
    shape_costs = (n_new_edges,)
    ds_costs = g_out.require_dataset('costs', dtype='float32',
                                     shape=shape_costs, chunks=shape_costs)
    ds_costs.n_threads = n_threads
    ds_costs[:] = new_costs

    return n_new_edges


def reduce_problem(graph_path,
                   costs_path,
                   scale,
                   config_path,
                   tmp_folder,
                   cost_accumulation="sum"):
    t0 = time.time()
    with open(config_path) as f:
        config = json.load(f)
        n_threads = config['n_threads']
        initial_block_shape = config['block_shape']
        roi = config.get('roi', None)

    # get the number of nodes and uv-ids at this scale level
    # as well as the initial node labeling
    shape = z5py.File(graph_path).attrs['shape']
    if scale == 0:
        f_graph = z5py.File(os.path.join(graph_path, 'graph'), use_zarr_format=False)
        initial_node_labeling = None
        n_nodes = f_graph.attrs['numberOfNodes']
        uv_ids = f_graph['edges'][:]
    else:
        f_problem = z5py.File(os.path.join(tmp_folder, 'merged_graph.n5/s%i' % scale))
        n_nodes = f_problem.attrs['numberOfNodes']
        uv_ids = f_problem['edges'][:]
        initial_node_labeling = f_problem['nodeLabeling'][:]
    n_edges = len(uv_ids)

    # get the costs
    costs = z5py.File(costs_path)['costs'][:]
    assert len(costs) == n_edges, "%i, %i" (len(costs), n_edges)

    # get the new node assignment
    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]
    blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)

    if roi is None:
        block_list = range(blocking.numberOfBlocks)
    else:
        block_list = blocking.getBlockIdsOverlappingBoundingBox(roi[0], roi[1],
                                                                [0, 0, 0]).tolist()

    n_new_nodes, node_labeling, new_initial_node_labeling = merge_nodes(tmp_folder,
                                                                        scale,
                                                                        block_list,
                                                                        n_nodes,
                                                                        uv_ids,
                                                                        initial_node_labeling,
                                                                        n_threads)
    # get the new edge assignment
    new_uv_ids, edge_labeling, new_costs = get_new_edges(uv_ids, node_labeling,
                                                         costs, cost_accumulation, n_threads)

    # FIXME this segfaults if we process a roi
    # serialize the input graph and costs for the next scale level
    n_new_edges = serialize_new_problem(graph_path, n_new_nodes, new_uv_ids,
                                        node_labeling, edge_labeling,
                                        new_costs, new_initial_node_labeling,
                                        shape, scale, initial_block_shape,
                                        tmp_folder, n_threads, roi)
    res_file = os.path.join(tmp_folder, 'reduce_problem_s%i.log' % scale)
    with open(res_file, 'w') as f:
        json.dump({'t': time.time() - t0,
                   'new_nodes': n_new_nodes,
                   'new_edges': n_new_edges}, f)

    print("Reduced graph from", n_nodes, "to", n_new_nodes, "nodes;",
          n_edges, "to", n_new_edges, "edges.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("costs_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("config_path", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()

    reduce_problem(args.graph_path,
                   args.costs_path,
                   args.scale,
                   args.config_path,
                   args.tmp_folder)
