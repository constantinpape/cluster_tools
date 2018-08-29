#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class BlockNodeLabelsBase(luigi.Task):
    """ BlockNodeLabels base class
    """

    task_name = 'block_node_labels'
    src_file = os.path.abspath(__file__)

    # input volumes and graph
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    # TODO
    # this belongs to the merge labels task
    # @staticmethod
    # def default_task_config():
    #     # we use this to get also get the common default config
    #     config = LocalTask.default_task_config()
    #     config.update({'ignore_label_gt': True})
    #     return config

    def clean_up_for_retry(self, block_list):
        # TODO does this work with the mixin pattern?
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'labels_path': self.labels_path, 'labels_key': self.labels_key,
                       'input_path': self.input_path,
                       'input_key': self.input_key,
                       'block_shape': block_shape,
                       'output_path': self.output_path, 'output_key': self.output_key})

        # make graph file and write shape as attribute
        shape = vu.get_shape(self.labels_path, self.labels_key)

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        n_jobs = min(len(block_list), self.max_jobs)

        # TODO initial implementation without proper parallelization.
        # remove this once we have scalable implementation
        n_jobs = 1

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class BlockNodeLabelsLocal(BlockNodeLabelsBase, LocalTask):
    """ BlockNodeLabels on local machine
    """
    pass


class BlockNodeLabelsSlurm(BlockNodeLabelsBase, SlurmTask):
    """ BlockNodeLabels on slurm cluster
    """
    pass


class BlockNodeLabelsLSF(BlockNodeLabelsBase, LSFTask):
    """ BlockNodeLabels on lsf cluster
    """
    pass


#
# Implementation
#


def _labels_for_block(block_id, blocking, ds_labels, ds_in):
    # read labels and input in this block
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    labels = ds_labels[bb]
    # check if label block is empty
    if (labels != 0).sum() == 0:
        return None
    input_ = ds_in[bb]
    # get the overlaps
    overlaps = ndist.computeLabelOverlaps(labels, input_)
    return overlaps


def block_node_labels(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    labels_path = config['labels_path']
    labels_key = config['labels_key']
    input_path = config['input_path']
    input_key = config['input_key']
    block_shape = config['block_shape']
    block_list = config['block_list']
    output_path = config['output_path']
    output_key = config['output_key']

    with vu.file_reader(labels_path, 'r') as f:
        shape = f[labels_key].shape

    # blocking = nt.blocking([0, 0, 0],
    #                        list(shape),
    #                        list(block_shape))

    # TODO change to proper blocking once we have scalable implementation
    blocking = nt.blocking([0, 0, 0],
                           list(block_shape),
                           list(block_shape))
    block_list = [0]

    results = [_labels_for_block(block_id, blocking,
                                 labels_path, labels_key,
                                 graph_path,
                                 input_path, input_key) for block_id in block_list]
    results = [res for res in results if res is not None]

    # check if we have results
    if len(results) == 0:
        raise RuntimeError("Woo")
        overlaps = {}

    elif len(results) == 1:
        overlaps = results[0]

    else:
        raise NotImplementedError("Scalable implementation not available")
        # merge the overlaps, we use results[0] as target vector
        overlaps = results[0]
        # TODO this looks sloooow, speed this up
        for ovlp in results[1:]:
            for node, node_ovlp in ovlp.items():
                if node in overlaps:
                    pass
                else:
                    overlaps[node] = node_ovlp

    # TODO properly save sub results for scalable implementation
    node_ids = np.array(overlaps.keys())
    node_max = node_ids.max()
    overlap_vector = np.zeros(node_max + 1, dtype='uint64')

    for node, ovlp in overlaps:
        ovl_labels, ovl_counts = np.array(ovlp.keys()), np.array(ovlp.values())
        max_ol = ovl_labels[np.argmax(ovl_counts)]
        overlap_vector[node] = max_ol

    with vu.file_reader(output_path) as f:
        f.create_dataset(output_key, data=overlap_vector, chunks=overlap_vector.shape)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    block_node_labels(job_id, path)
