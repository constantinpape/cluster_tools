#! /usr/bin/python

import os
import sys
import json

import luigi
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Stitching Tasks
#

class SimpleStitchEdgesBase(luigi.Task):
    """ SimpleStitchEdges base class
    """

    task_name = 'simple_stitch_edges'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    graph_path = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        shape = vu.get_shape(self.labels_path, self.labels_key)
        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        graph_key = 's0/graph'
        with vu.file_reader(self.graph_path, 'r') as f:
            n_edges = f[graph_key].attrs['numberOfEdges']

        config = self.get_task_config()
        tmp_file = os.path.join(self.tmp_folder, 'stitch_edges.n5')
        config.update({'out_path': tmp_file,
                       'graph_path': self.graph_path,
                       'labels_path': self.labels_path,
                       'labels_key': self.labels_key,
                       'n_edges': n_edges})

        with vu.file_reader(tmp_file) as f:
            f.require_group('job_results')

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class SimpleStitchEdgesLocal(SimpleStitchEdgesBase, LocalTask):
    """
    SimpleStitchEdges on local machine
    """
    pass


class SimpleStitchEdgesSlurm(SimpleStitchEdgesBase, SlurmTask):
    """
    SimpleStitchEdges on slurm cluster
    """
    pass


class SimpleStitchEdgesLSF(SimpleStitchEdgesBase, LSFTask):
    """
    SimpleStitchEdges on lsf cluster
    """
    pass


def simple_stitch_edges(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    graph_path = config['graph_path']
    labels_path = config['labels_path']
    labels_key = config['labels_key']
    n_edges = config['n_edges']
    block_list = config['block_list']

    out_path = config['out_path']
    out_key = 'job_results/job_%i' % job_id

    block_prefix = os.path.join(graph_path, 's0/sub_graphs/block_')
    res = ndist.find1DEdges(block_prefix, labels_path, labels_key, n_edges, block_list)

    with vu.file_reader(out_path) as f:
        f.create_dataset(out_key, data=res, compression='gzip')

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    simple_stitch_edges(job_id, path)
