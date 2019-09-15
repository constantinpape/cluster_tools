#! /bin/python

import os
import sys
import json
from concurrent import futures

import numpy as np
import vigra
import luigi
import nifty
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.segmentation_utils as su
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Lifted Multicut Tasks
#


class ResolveIndividualObjectsBase(luigi.Task):
    """ ResolveIndividualObjects base class
    """

    task_name = 'resolve_inidividual_objects'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    problem_path = luigi.Parameter()
    objects_path = luigi.Parameter()
    objects_group = luigi.Parameter()
    assignment_in_path = luigi.Parameter()
    assignment_in_key = luigi.Parameter()
    assignment_out_path = luigi.Parameter()
    assignment_out_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'agglomerator': 'kernighan-lin',
                       'time_limit_solver': None})
        return config

    def run_impl(self):
        shebang = self.global_config_values()[0]
        self.init(shebang)

        config = self.get_task_config()
        config.update({'problem_path': self.problem_path,
                       'objects_path': self.objects_path,
                       'objects_group': self.objects_group,
                       'assignment_in_path': self.assignment_in_path,
                       'assignment_in_key': self.assignment_in_key,
                       'assignment_out_path': self.assignment_out_path,
                       'assignment_out_key': self.assignment_out_key})

        n_jobs = 1
        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class ResolveIndividualObjectsLocal(ResolveIndividualObjectsBase, LocalTask):
    """ ResolveIndividualObjects on local machine
    """
    pass


class ResolveIndividualObjectsSlurm(ResolveIndividualObjectsBase, SlurmTask):
    """ ResolveIndividualObjects on slurm cluster
    """
    pass


class ResolveIndividualObjectsLSF(ResolveIndividualObjectsBase, LSFTask):
    """ ResolveIndividualObjects on lsf cluster
    """
    pass


#
# Implementation
#

def _solve_objects(objects, graph, assignments, costs,
                   agglomerator, n_threads, time_limit):
    lifted_uv_ds = objects['uvs']
    lifted_costs_ds = objects['costs']

    uv_ids = graph.uvIds()

    n_objects = lifted_uv_ds.shape[0]
    assert lifted_costs_ds.shape[0] == n_objects

    def solve_object(obj_id):

        # try to load the object's edges and costs and continue if not present
        lifted_uvs = lifted_uv_ds.read_chunk((obj_id,))
        if lifted_uvs is None:
            return None
        n_lifted_edges = len(lifted_uvs) // 2
        lifted_uvs = lifted_uvs.reshape((n_lifted_edges, 2))
        lifted_costs = lifted_costs_ds.read_chunk((obj_id,))
        assert lifted_costs is not None
        assert len(lifted_costs) == len(lifted_uvs), "%i, %i" % (len(lifted_costs),
                                                                 len(lifted_uvs))

        # get node ids for this object
        obj_mask = assignments == obj_id
        node_ids = np.where(obj_mask)[0].astype('uint64')
        inner_edges, _ = graph.extractSubgraphFromNodes(node_ids)

        sub_uvs = uv_ids[inner_edges]
        sub_costs = costs[inner_edges]
        assert len(sub_uvs) == len(sub_costs)

        # relabel all consecutive
        nodes_relabeled, max_id, mapping = vigra.analysis.relabelConsecutive(node_ids,
                                                                             start_label=0,
                                                                             keep_zeros=False)
        sub_uvs = nt.takeDict(mapping, sub_uvs)
        lifted_uvs = nt.takeDict(mapping, lifted_uvs)

        n_local_nodes = max_id + 1
        sub_graph = nifty.graph.undirectedGraph(n_local_nodes)
        sub_graph.insertEdges(sub_uvs)

        sub_assignments = agglomerator(sub_graph, sub_costs, lifted_uvs, lifted_costs,
                                       time_limit=time_limit)
        vigra.analysis.relabelConsecutive(sub_assignments, out=sub_assignments,
                                          start_label=1, keep_zeros=False)
        return obj_mask, sub_assignments

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(solve_object, obj_id) for obj_id in range(n_objects)]
        results = [t.result() for t in tasks]

    # filter results
    results = [res for res in results if res is not None]

    # could e vectorized and parallelized if necessary
    for obj_mask, sub_assignments in results:
        max_id = int(assignments.max()) + 1
        sub_assignments += max_id
        assignments[obj_mask] = sub_assignments

    return assignments


def resolve_inidividual_objects(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    # input configs
    problem_path = config['problem_path']

    objects_path = config['objects_path']
    objects_group = config['objects_group']
    assignment_in_path = config['assignment_in_path']
    assignment_in_key = config['assignment_in_key']
    assignment_out_path = config['assignment_out_path']
    assignment_out_key = config['assignment_out_key']

    agglomerator_key = config['agglomerator']
    time_limit = config.get('time_limit_solver', None)
    n_threads = config.get('threads_per_job', 1)

    fu.log("reading problem from %s" % problem_path)
    problem = vu.file_reader(problem_path)

    # load the costs
    costs_key = 's0/costs'
    fu.log("reading costs from path in problem: %s" % costs_key)
    ds = problem[costs_key]
    ds.n_threads = n_threads
    costs = ds[:]

    # load the graph
    graph_key = 's0/graph'
    fu.log("reading graph from path in problem: %s" % graph_key)
    graph = ndist.Graph(os.path.join(problem_path, graph_key),
                        numberOfThreads=n_threads)

    fu.log("using agglomerator %s" % agglomerator_key)
    agglomerator = su.key_to_lifted_agglomerator(agglomerator_key)

    # load assignments
    f = vu.file_reader(assignment_in_path)
    ds = f[assignment_in_key]
    ds.n_threads = n_threads
    assignments = ds[:]

    # load the object group
    objects = vu.file_reader(objects_path)[objects_group]
    new_assignments = _solve_objects(objects, graph, assignments, costs,
                                     agglomerator, n_threads, time_limit)

    chunks = ds.chunks
    f = vu.file_reader(assignment_out_path)
    ds = f.require_dataset(assignment_out_key, shape=new_assignments.shape, chunks=chunks,
                           compression='gzip', dtype='uint64')
    ds.n_threads = n_threads
    ds[:] = new_assignments

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    resolve_inidividual_objects(job_id, path)
