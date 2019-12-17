#! /bin/python

import os
import sys
import json
import numpy as np
import luigi

import vigra
import nifty

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Graph Watershed Tasks
#

class GraphWatershedAssignmentsBase(luigi.Task):
    """ GraphWatershedAssignments base class
    """

    task_name = 'graph_watershed_assignments'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    problem_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    features_key = luigi.Parameter()
    filter_nodes_path = luigi.Parameter()
    #
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    relabel = luigi.BoolParameter(default=False)
    from_costs = luigi.BoolParameter(default=True)
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'assignment_path': self.assignment_path,
                       'assignment_key': self.assignment_key,
                       'problem_path': self.problem_path,
                       'graph_key': self.graph_key,
                       'features_key': self.features_key,
                       'filter_nodes_path': self.filter_nodes_path,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'from_costs': self.from_costs,
                       'relabel': self.relabel})

        n_jobs = 1
        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class GraphWatershedAssignmentsLocal(GraphWatershedAssignmentsBase, LocalTask):
    """ GraphWatershedAssignments on local machine
    """
    pass


class GraphWatershedAssignmentsSlurm(GraphWatershedAssignmentsBase, SlurmTask):
    """ GraphWatershedAssignments on slurm cluster
    """
    pass


class GraphWatershedAssignmentsLSF(GraphWatershedAssignmentsBase, LSFTask):
    """ GraphWatershedAssignments on lsf cluster
    """
    pass


#
# Implementation
#

def graph_watershed_assignments(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    # load from config
    assignment_path = config['assignment_path']
    assignment_key = config['assignment_key']
    problem_path = config['problem_path']
    graph_key = config['graph_key']
    features_key = config['features_key']
    filter_nodes_path = config['filter_nodes_path']
    output_path = config['output_path']
    output_key = config['output_key']
    relabel = config['relabel']
    from_costs = config['from_costs']
    n_threads = config.get('threads_per_job', 1)

    # load the uv-ids, features and assignments
    fu.log("Read features and edges from %s" % problem_path)
    with vu.file_reader(problem_path, 'r') as f:
        ds = f['%s/edges' % graph_key]
        ds.n_threads = n_threads
        uv_ids = ds[:]
        n_nodes = int(uv_ids.max()) + 1

        ds = f[features_key]
        ds.n_threads = n_threads
        if ds.ndim == 2:
            features = ds[:, 0].squeeze()
        else:
            features = ds[:]

    if from_costs:
        minc = features.min()
        fu.log("Mapping costs with range %f to %f to range 0 to 1" % (minc, features.max()))
        features -= minc
        features /= features.max()
        features = 1. - features

    fu.log("Read assignments from %s" % assignment_path)
    with vu.file_reader(assignment_path, 'r') as f:
        ds = f[assignment_key]
        ds.n_threads = n_threads
        chunks = ds.chunks
        assignments = ds[:]
    assert n_nodes == len(assignments),\
        "Expected number of nodes %i and number of assignments %i does not agree" % (n_nodes, len(assignments))

    seed_offset = int(assignments.max()) + 1

    # load the discard ids
    discard_ids = np.load(filter_nodes_path)
    assert 0 not in discard_ids, "Breaks logic"

    # build the new graph
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)

    # run graph watershed to get the new assignments
    # map zero label to new id
    assignments[assignments == 0] = seed_offset

    discard_mask = np.in1d(assignments, discard_ids)
    assignments[discard_mask] = 0

    n_discard = int(discard_mask.sum())
    fu.log("Discarding %i / %i fragments" % (n_discard, assignments.size))
    fu.log("Start grah watershed")
    assignments = nifty.graph.edgeWeightedWatershedsSegmentation(graph, assignments, features)
    fu.log("Finished graph watershed")
    assignments[assignments == seed_offset] = 0

    if relabel:
        max_id = vigra.analysis.relabelConsecutive(assignments, out=assignments,
                                                   start_label=1, keep_zeros=True)[1]
        fu.log("Max-id after relabeling: %i (before was %i)" % (max_id, seed_offset - 1))

    with vu.file_reader(output_path) as f:
        ds = f.require_dataset(output_key, shape=assignments.shape, chunks=chunks,
                               compression='gzip', dtype='uint64')
        ds.n_threads = n_threads
        ds[:] = assignments

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    graph_watershed_assignments(job_id, path)
