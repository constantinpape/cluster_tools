#! /usr/bin/python

import os
import json
import sys
import luigi

import numpy as np
import nifty
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.volume_utils as vu
from elf.segmentation.multicut import get_multicut_solver, transform_probabilities_to_costs
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class StitchingMulticutBase(luigi.Task):
    task_name = 'stitching_multicut'
    src_file = os.path.abspath(__file__)

    problem_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    features_key = luigi.Parameter()
    edge_key = luigi.Parameter()

    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'beta1': 0.5, 'beta2': 0.75, 'time_limit_solver': None,
                       'agglomerator': 'kernighan-lin'})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the stitching config
        task_config = self.get_task_config()

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'problem_path': self.problem_path, 'graph_key': self.graph_key,
                            'features_key': self.features_key, 'edge_key': self.edge_key,
                            'output_path': self.output_path, 'output_key': self.output_key})

        # prime and run the jobs
        n_jobs = 1
        self.prepare_jobs(n_jobs, None, task_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class StitchingMulticutLocal(StitchingMulticutBase, LocalTask):
    pass


class StitchingMulticutSlurm(StitchingMulticutBase, SlurmTask):
    pass


class StitchingMulticutLSF(StitchingMulticutBase, LSFTask):
    pass


#
# Implementation
#


def stitching_multicut(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    problem_path = config['problem_path']
    graph_key = config['graph_key']
    features_key = config['features_key']
    edge_key = config['edge_key']

    output_path = config['output_path']
    output_key = config['output_key']

    beta1 = config['beta1']
    beta2 = config['beta2']
    n_threads = config.get('threads_per_job', 1)
    time_limit = config.get('time_limit_solver', None)
    agglomerator_key = config.get('agglomerator', 'kernighan-lin')

    # load edges and features
    fu.log("Loading features and edges")
    with vu.file_reader(problem_path, 'r') as f:
        ds = f[features_key]
        ds.n_threads = n_threads
        feats = ds[:]

        ds = f[edge_key]
        ds.n_threads = n_threads
        stitch_edges = ds[:].astype('bool')

        g = f[graph_key]
        ds = g['edges']
        ds.n_threads = n_threads
        uv_ids = ds[:]

    # load graph
    fu.log("Building graph")
    n_nodes = int(uv_ids.max()) + 1
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    n_edges = graph.numberOfEdges

    # compute costs
    fu.log("Computing costs")
    feats, sizes = feats[:, 0], feats[:, -1]
    costs = np.zeros(n_edges, dtype='float32')
    costs[stitch_edges] = transform_probabilities_to_costs(feats[stitch_edges],
                                                           edge_sizes=sizes[stitch_edges],
                                                           beta=beta1)
    costs[~stitch_edges] = transform_probabilities_to_costs(feats[~stitch_edges],
                                                            edge_sizes=sizes[~stitch_edges],
                                                            beta=beta2)

    # solve multicut
    solver = get_multicut_solver(agglomerator_key)
    fu.log("Start multicut with solver %s" % agglomerator_key)
    if time_limit is not None:
        fu.log("With time limit %i s" % time_limit)
    node_labels = solver(graph, costs,
                         n_threads=n_threads, time_limit=time_limit)
    fu.log("Multicut done")

    # write result
    with vu.file_reader(output_path) as f:
        chunks = (min(int(1e6), len(node_labels)),)
        ds = f.require_dataset(output_key, shape=node_labels.shape, chunks=chunks,
                               compression='gzip', dtype=node_labels.dtype)
        ds.n_threads = n_threads
        ds[:] = node_labels

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    stitching_multicut(job_id, path)
