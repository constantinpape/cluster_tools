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

class MergeMorphologyBase(luigi.Task):
    """ MergeMorphology base class
    """

    task_name = 'merge_morphology'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    number_of_labels = luigi.IntParameter()
    prefix = luigi.Parameter()
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

        out_shape = (int(self.number_of_labels), 11)
        out_chunks = (min(int(self.number_of_labels), 100000), 11)
        block_list = vu.blocks_in_volume([out_shape[0]], [out_chunks[0]])

        # create output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=out_shape,
                              chunks=out_chunks, compression='gzip',
                              dtype='float64')

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'out_shape': out_shape,
                       'out_chunks': out_chunks})

        # prime and run the jobs
        self.prepare_jobs(self.max_jobs, block_list, config, self.prefix)
        self.submit_jobs(self.max_jobs, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.prefix)
        self.check_jobs(self.max_jobs, self.prefix)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.prefix))


class MergeMorphologyLocal(MergeMorphologyBase, LocalTask):
    """ MergeMorphology on local machine
    """
    pass


class MergeMorphologySlurm(MergeMorphologyBase, SlurmTask):
    """ MergeMorphology on slurm cluster
    """
    pass


class MergeMorphologyLSF(MergeMorphologyBase, LSFTask):
    """ MergeMorphology on lsf cluster
    """
    pass


#
# Implementation
#


def merge_morphology(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    block_list = config['block_list']
    out_shape = config['out_shape']
    out_chunks = config['out_chunks']

    blocking = nt.blocking([0], out_shape[:1], out_chunks[:1])

    # merge and serialize the overlaps
    for block_id in block_list:
        block = blocking.getBlock(block_id)
        label_begin = block.begin[0]
        label_end = block.end[0]
        ndist.mergeAndSerializeMorphology(os.path.join(input_path, input_key),
                                          os.path.join(output_path, output_key),
                                          labelBegin=label_begin, labelEnd=label_end)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_morphology(job_id, path)
