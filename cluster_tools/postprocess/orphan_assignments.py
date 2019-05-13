#! /bin/python

import os
import sys
import json
import numpy as np
import luigi

import vigra
import nifty
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Orphan Filter Tasks
#

class OrphanAssignmentsBase(luigi.Task):
    """ OrphanAssignments base class
    """

    task_name = 'orphan_assignments'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    relabel = luigi.BoolParameter(default=False)
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
                       'graph_path': self.graph_path,
                       'graph_key': self.graph_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'relabel': self.relabel})

        n_jobs = 1
        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class OrphanAssignmentsLocal(OrphanAssignmentsBase, LocalTask):
    """ OrphanAssignments on local machine
    """
    pass


class OrphanAssignmentsSlurm(OrphanAssignmentsBase, SlurmTask):
    """ OrphanAssignments on slurm cluster
    """
    pass


class OrphanAssignmentsLSF(OrphanAssignmentsBase, LSFTask):
    """ OrphanAssignments on lsf cluster
    """
    pass


#
# Implementation
#

def orphan_assignments(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    # load from config
    assignment_path = config['assignment_path']
    assignment_key = config['assignment_key']
    graph_path = config['graph_path']
    graph_key = config['graph_key']
    output_path = config['output_path']
    output_key = config['output_key']
    relabel = config['relabel']
    n_threads = config.get('threads_per_job', 1)

    # load the uv-ids and assignments
    with vu.file_reader(graph_path) as f:
        ds = f['%s/edges' % graph_key]
        ds.n_threads = n_threads
        uv_ids = ds[:]
    with vu.file_reader(assignment_path) as f:
        ds = f[assignment_key]
        ds.n_threads = n_threads
        chunks = ds.chunks
        assignments = ds[:]

    n_new_nodes = int(assignments.max()) + 1
    # find the new uv-ids
    edge_mapping = nt.EdgeMapping(uv_ids, assignments, numberOfThreads=n_threads)
    new_uv_ids = edge_mapping.newUvIds()

    # find all orphans = segments that have node degree one
    ids, node_degrees = np.unique(new_uv_ids, return_counts=True)
    orphans = ids[node_degrees == 1]
    n_orphans = len(orphans)
    fu.log("Found %i orphans of %i clusters" % (n_orphans, n_new_nodes))

    # make graph for fast neighbor search
    graph = nifty.graph.undirectedGraph(n_new_nodes)
    graph.insertEdges(new_uv_ids)

    orphan_assignments = np.array([next(graph.nodeAdjacency(orphan_id))[0]
                                   for orphan_id in orphans],)
    assert len(orphan_assignments) == n_orphans, "%i, %i" % (len(orphan_assignments), n_orphans)
    assignments[orphans] = orphan_assignments.astype('uint64')

    if relabel:
        vigra.analysis.relabelConsecutive(assignments, out=assignments,
                                          start_label=1, keep_zeros=True)

    with vu.file_reader(output_path) as f:
        ds = f.require_dataset(output_key, shape=assignments.shape, chunks=chunks,
                               compression='gzip', dtype='uint64')
        ds[:] = assignments

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    orphan_assignments(job_id, path)
