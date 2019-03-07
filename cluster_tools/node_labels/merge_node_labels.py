#! /bin/python

import os
import sys
import json
import numpy as np
from concurrent import futures

import luigi
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class MergeNodeLabelsBase(luigi.Task):
    """ MergeNodeLabels base class
    """

    task_name = 'merge_node_labels'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    max_overlap = luigi.BoolParameter(default=True)
    ignore_label = luigi.IntParameter(default=None)
    serialize_counts = luigi.BoolParameter(default=False)
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

        with vu.file_reader(self.input_path, 'r') as f:
            number_of_labels = int(f[self.input_key].attrs['maxId']) + 1

        node_shape = (number_of_labels,)
        node_chunks = (min(number_of_labels, 100000),)
        block_list = vu.blocks_in_volume(node_shape, node_chunks)

        # create output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=node_shape,
                              chunks=node_chunks, compression='gzip',
                              dtype='uint64')

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'max_overlap': self.max_overlap,
                       'node_shape': node_shape,
                       'node_chunks': node_chunks,
                       'ignore_label': self.ignore_label,
                       'serialize_counts': self.serialize_counts})

        # prime and run the jobs
        prefix = 'max_ol' if self.max_overlap else 'all_ol'
        n_jobs = min(self.max_jobs, len(block_list))
        self.prepare_jobs(n_jobs, block_list, config, prefix)
        self.submit_jobs(n_jobs, prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(prefix)
        self.check_jobs(n_jobs, prefix)

    # part of the luigi API
    def output(self):
        max_ol_str = 'max_ol' if self.max_overlap else 'all_ol'
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % max_ol_str))


class MergeNodeLabelsLocal(MergeNodeLabelsBase, LocalTask):
    """ MergeNodeLabels on local machine
    """
    pass


class MergeNodeLabelsSlurm(MergeNodeLabelsBase, SlurmTask):
    """ MergeNodeLabels on slurm cluster
    """
    pass


class MergeNodeLabelsLSF(MergeNodeLabelsBase, LSFTask):
    """ MergeNodeLabels on lsf cluster
    """
    pass


#
# Implementation
#


def merge_node_labels(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    max_overlap = config['max_overlap']
    n_threads = config.get('threads_per_job', 1)
    block_list = config['block_list']
    node_shape = config['node_shape']
    node_chunks = config['node_chunks']
    ignore_label = config['ignore_label']
    serialize_counts = config['serialize_counts']

    blocking = nt.blocking([0], list(node_shape), list(node_chunks))

    # merge and serialize the overlaps
    for block_id in block_list:
        block = blocking.getBlock(block_id)
        label_begin = block.begin[0]
        label_end = block.end[0]
        ndist.mergeAndSerializeOverlaps(os.path.join(input_path, input_key),
                                        os.path.join(output_path, output_key),
                                        max_overlap=max_overlap,
                                        labelBegin=label_begin, labelEnd=label_end,
                                        numberOfThreads=n_threads,
                                        ignoreLabel=0 if ignore_label is None else ignore_label,
                                        serializeCount=serialize_counts)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_node_labels(job_id, path)
