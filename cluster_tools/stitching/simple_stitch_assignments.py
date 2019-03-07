#! /usr/bin/python

import os
import sys
import json

import luigi
import nifty.ufd as nufd
import numpy as np

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Stitching Tasks
#

class SimpleStitchAssignmentsBase(luigi.Task):
    """ SimpleStitchAssignments base class
    """

    task_name = 'simple_stitch_assignments'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    problem_path = luigi.Parameter()
    features_key = luigi.Parameter()
    graph_key = luigi.Parameter()
    assignments_path = luigi.Parameter()
    assignments_key = luigi.Parameter()
    edge_size_threshold = luigi.IntParameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        with vu.file_reader(self.problem_path, 'r') as f:
            shape = f.attrs['shape']
        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        config = self.get_task_config()
        tmp_file = os.path.join(self.tmp_folder, 'stitch_edges.n5')
        config.update({'input_path': tmp_file,
                       'problem_path': self.problem_path,
                       'features_key': self.features_key,
                       'graph_key': self.graph_key,
                       'assignments_path': self.assignments_path,
                       'assignments_key': self.assignments_key,
                       'edge_size_threshold': self.edge_size_threshold,
                       'n_jobs': n_jobs})

        with vu.file_reader(tmp_file) as f:
            f.require_group('job_results')

        # we only have a single job to find the labeling
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(1)


class SimpleStitchAssignmentsLocal(SimpleStitchAssignmentsBase, LocalTask):
    """
    SimpleStitchAssignments on local machine
    """
    pass


class SimpleStitchAssignmentsSlurm(SimpleStitchAssignmentsBase, SlurmTask):
    """
    SimpleStitchAssignments on slurm cluster
    """
    pass


class SimpleStitchAssignmentsLSF(SimpleStitchAssignmentsBase, LSFTask):
    """
    SimpleStitchAssignments on lsf cluster
    """
    pass


def simple_stitch_assignments(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    input_path = config['input_path']
    problem_path = config['problem_path']
    features_key = config['features_key']
    graph_key = config['graph_key']
    assignments_path = config['assignments_path']
    assignments_key = config['assignments_key']
    edge_size_threshold = config['edge_size_threshold']
    n_jobs = config['n_jobs']

    # load the edge results of the first
    f = vu.file_reader(input_path, 'r')
    key0 = 'job_results/job_0'
    merge_edges = f[key0][:].astype('bool')

    for job in range(1, n_jobs):
        key = 'job_results/job_%i' % job
        res_job = f[key][:].astype('bool')
        merge_edges = np.logical_and(merge_edges, res_job)

    # load edge sizes to not merge across tiny edges
    f = vu.file_reader(problem_path, 'r')
    edge_sizes = f[features_key][:, -1].squeeze()
    merge_edges = np.logical_and(merge_edges, edge_sizes > edge_size_threshold)

    # load the uv-ids
    uv_key = '%s/%s' % (graph_key, 'edges')
    uv_ids = f[uv_key][:]
    assert len(uv_ids) == len(merge_edges)

    n_nodes = int(uv_ids.max()) + 1
    merge_ids = uv_ids[merge_edges]

    ufd = nufd.ufd(n_nodes)
    ufd.merge(merge_ids)

    node_labeling = ufd.elementLabeling()
    # TODO relabel instead of assertion
    assert node_labeling[0] == 0

    with vu.file_reader(assignments_path) as f:
        chunks = (min(int(1e5), len(node_labeling)),)
        ds = f.require_dataset(assignments_key, shape=node_labeling.shape, compression='gzip',
                               chunks=chunks, dtype='uint64')
        ds[:] = node_labeling

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    simple_stitch_assignments(job_id, path)
