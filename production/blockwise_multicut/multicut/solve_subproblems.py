#! /usr/bin/python

import time
import os
import argparse
import subprocess
import json
from concurrent import futures
import numpy as np

import z5py
import nifty
import nifty.distributed as ndist
import luigi
import cremi_tools.segmentation as cseg

# TODO support more agglomerators
AGGLOMERATORS = {"multicut_kl": cseg.Multicut("kernighan-lin")}


class SolveSubproblemTask(luigi.Task):
    """
    Compute initial sub-graphs
    """

    graph_path = luigi.Parameter()
    costs_path = luigi.Parameter()
    scale = luigi.IntParameter()
    max_jobs = luigi.IntParameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def _prepare_jobs(self, n_jobs, n_blocks, block_shape, n_threads):
        block_list = list(range(n_blocks))
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_shape': block_shape,
                          'n_threads': n_threads,
                          'block_list': block_jobs}
            config_path = os.path.join(self.tmp_folder,
                                       'solve_subproblems_config_s%i_job%i.json' % (self.scale,
                                                                                    job_id))
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id, n_threads):
        script_path = os.path.join(self.tmp_folder, 'solve_subproblems.py')
        config_path = os.path.join(self.tmp_folder,
                                   'solve_subproblems_config_s%i_job%i.json' % (self.scale,
                                                                                job_id))
        command = '%s %s %s %i %i %s %s' % (script_path, self.graph_path,
                                            self.costs_path, self.scale,
                                            job_id, config_path,
                                            self.tmp_folder)
        log_file = os.path.join(self.tmp_folder,
                                'logs', 'log_solve_subproblems_s%i_%i' % (self.scale, job_id))
        err_file = os.path.join(self.tmp_folder,
                                'error_logs', 'err_solve_subproblems_s%i_%i.err' % (self.scale,
                                                                                    job_id))
        bsub_command = ('bsub -n %i -J solve_subproblems_%i ' % (n_threads, job_id) +
                        '-We %i -o %s -e %s \'%s\'' % (self.time_estimate,
                                                       log_file, err_file, command))
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _collect_outputs(self, n_blocks):
        times = []
        processed_blocks = []
        for block_id in range(n_blocks):
            res_file = os.path.join(self.tmp_folder,
                                    'subproblem_s%i_%i.log' % (self.scale, block_id))
            try:
                with open(res_file) as f:
                    res = json.load(f)
                times.append(res['t'])
                processed_blocks.append(block_id)
                os.remove(res_file)
            except Exception:
                continue
        return processed_blocks, times

    def run(self):
        from production import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'solve_subproblems.py'),
                              os.path.join(self.tmp_folder, 'solve_subproblems.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            initial_block_shape = config['block_shape']
            n_threads = config['n_threads']
            # TODO support computation with roi
            if 'roi' in config:
                roi = config['roi']
            else:
                roi = None

        # get number of blocks
        factor = 2**self.scale
        block_shape = [factor * bs for bs in initial_block_shape]
        shape = z5py.File(self.graph_path).attrs['shape']
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks
        # find the actual number of jobs and prepare job configs
        n_jobs = min(n_blocks, self.max_jobs)
        self._prepare_jobs(n_jobs, n_blocks, initial_block_shape, n_threads)

        # submit the jobs
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(n_jobs) as tp:
                tasks = [tp.submit(self._submit_job, job_id, n_threads)
                         for job_id in range(n_jobs)]
                [t.result() for t in tasks]
        else:
            for job_id in range(n_jobs):
                self._submit_job(job_id)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        # check the job outputs
        processed_blocks, times = self._collect_outputs(n_blocks)
        assert len(processed_blocks) == len(times)
        success = len(processed_blocks) == n_blocks

        # write output file if we succeed, otherwise write partial
        # success to different file and raise exception
        if success:
            out = self.output()
            # TODO does 'out' support with job?
            fres = out.open('w')
            json.dump({'times': times}, fres)
            fres.close()
        else:
            log_path = os.path.join(self.tmp_folder, 'solve_subproblems_s%i_partial.json' % self.scale)
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("SolveSubproblemTask failed, "
                               "%i / %i blocks processed, "
                               "serialized partial results to %s" % (len(processed_blocks),
                                                                     n_blocks,
                                                                     log_path))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'solve_subproblems_s%i.log' % self.scale))


def solve_block_subproblem(block_id,
                           graph,
                           block_prefix,
                           costs,
                           agglomerator,
                           shape,
                           block_shape,
                           tmp_folder,
                           scale,
                           cut_outer_edges):

    t0 = time.time()
    # load the nodes in this sub-block and map them
    # to our current node-labeling
    block_path = block_prefix + str(block_id)
    assert os.path.exists(block_path), block_path
    nodes = ndist.loadNodes(block_path)

    # # the ignore label (== 0) spans a lot of blocks, hence it would slow down our
    # # subgraph extraction, which looks at all the blocks containing the node,
    # # enormously, so we skip it
    # # we make sure that these are cut later
    # if nodes[0] == 0:
    #     nodes = nodes[1:]

    # # if we have no nodes left after, we return none
    # if len(nodes) == 0:
    #     return None

    # # extract the local subgraph
    # inner_edges, outer_edges, sub_uvs = ndist.extractSubgraphFromNodes(nodes,
    #                                                                    block_prefix,
    #                                                                    shape,
    #                                                                    block_shape,
    #                                                                    block_id)
    inner_edges, outer_edges, sub_uvs = graph.extractSubgraphFromNodes(nodes)

    # if we had only a single node (i.e. no edge, return the outer edges)
    if len(nodes) == 1:
        cut_edge_ids = outer_edges if cut_outer_edges else []
        np.save(os.path.join(tmp_folder, 'subproblem_s%i_%i.npy' % (scale, block_id)),
                cut_edge_ids)
        res_path = os.path.join(tmp_folder, 'subproblem_s%i_%i.log' % (scale, block_id))
        with open(res_path, 'w') as f:
            json.dump({'t': time.time() - t0}, f)
        return

    assert len(sub_uvs) == len(inner_edges)
    assert len(sub_uvs) > 0, str(block_id)

    n_local_nodes = int(sub_uvs.max() + 1)
    sub_graph = nifty.graph.undirectedGraph(n_local_nodes)
    sub_graph.insertEdges(sub_uvs)

    sub_costs = costs[inner_edges]
    assert len(sub_costs) == sub_graph.numberOfEdges
    # print(len(sub_costs))

    sub_result = agglomerator(sub_graph, sub_costs)
    sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]

    assert len(sub_edgeresult) == len(inner_edges)
    cut_edge_ids = inner_edges[sub_edgeresult]

    # print("block", block_id, "number cut_edges:", len(cut_edge_ids))
    # print("block", block_id, "number outer_edges:", len(outer_edges))

    if cut_outer_edges:
        cut_edge_ids = np.concatenate([cut_edge_ids, outer_edges])

    np.save(os.path.join(tmp_folder, 'subproblem_s%i_%i.npy' % (scale, block_id)),
            cut_edge_ids)
    res_path = os.path.join(tmp_folder, 'subproblem_s%i_%i.log' % (scale, block_id))
    with open(res_path, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


def solve_subproblems(graph_path,
                      costs_path,
                      scale,
                      job_id,
                      config_path,
                      tmp_folder,
                      cut_outer_edges=True):

    # TODO support more agglomerators
    agglomerator_key = 'multicut_kl'
    agglomerator = AGGLOMERATORS[agglomerator_key]
    costs = z5py.File(costs_path)['costs'][:]

    with open(config_path) as f:
        config = json.load(f)
        initial_block_shape = config['block_shape']
        n_threads = config['n_threads']
        block_ids = config['block_list']

    shape = z5py.File(graph_path).attrs['shape']
    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    if scale == 0:
        graph_path_ = os.path.join(graph_path, 'graph')
        block_prefix = os.path.join(graph_path, 'sub_graphs', 's0', 'block_')
    else:
        graph_path_ = os.path.join(tmp_folder, 'merged_graph.n5', 's%i' % scale)
        block_prefix = os.path.join(graph_path_, 'sub_graphs', 'block_')

    # TODO parallelize ?!
    # load the complete graph
    graph = ndist.Graph(graph_path_)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(solve_block_subproblem,
                           block_id,
                           graph,
                           block_prefix,
                           costs,
                           agglomerator,
                           shape,
                           block_shape,
                           tmp_folder,
                           scale,
                           cut_outer_edges)
                 for block_id in block_ids]
        results = [t.result() for t in tasks]

    results = [res for res in results if res is not None]
    if len(results) > 0:
        cut_edge_ids = np.concatenate(results)
        cut_edge_ids = np.unique(cut_edge_ids).astype('uint64')
    else:
        cut_edge_ids = np.zeros(0, dtype='uint64')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("costs_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("job_id", type=int)
    parser.add_argument("config_path", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()

    solve_subproblems(args.graph_path,
                      args.costs_path,
                      args.scale,
                      args.job_id,
                      args.config_path,
                      args.tmp_folder)
