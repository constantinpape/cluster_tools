#! /usr/bin/python

import json
import os
import sys

import luigi
import nifty.distributed as ndist
from elf.io import open_file

import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.volume_utils as vu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


class LabelBlockMappingBase(luigi.Task):
    """ LabelBlockMapping base class
    """

    task_name = 'label_block_mapping'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    prefix = luigi.Parameter()
    # if unique path is given, we only compute the label block mapping for the unique ids
    # otherwise, number_of_labels has to be passed and we comput the labels in the range
    unique_path = luigi.Parameter(default='')
    unique_key = luigi.Parameter(default='')
    number_of_labels = luigi.IntParameter(default=None)
    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        config.update({'compression': 'gzip'})
        return config

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, _, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # shape and chunks for the id space
        ds_shape = (int(2**63 - 1),)  # open-ended shape
        # we use a chunk-size of 10k, but this could also be a parameter
        chunks = (10000,)

        config = self.get_task_config()
        # create the output dataset
        with vu.file_reader(self.output_path) as f:
            compression = config.get('compression', 'gzip')
            f.require_dataset(self.output_key, shape=ds_shape, compression=compression,
                              chunks=chunks, dtype='int8')

        config.update({"input_path": self.input_path, "input_key": self.input_key,
                       "output_path": self.output_path, "output_key": self.output_key})

        assert (self.number_of_labels is not None) != (self.unique_path != '')
        if self.number_of_labels is not None:
            config.update({"number_of_labels": self.number_of_labels})
        if self.unique_path != '':
            assert self.unique_key != ''
            config.update({'unique_path': self.unique_path, 'unique_key': self.unique_key})

        if roi_begin is not None:
            assert roi_end is not None
            config.update({'roi_begin': roi_begin, 'roi_end': roi_end})

        # prime and run the jobs
        self.prepare_jobs(1, None, config, self.prefix)
        self.submit_jobs(1, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.prefix)
        self.check_jobs(1, self.prefix)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.prefix))


class LabelBlockMappingLocal(LabelBlockMappingBase, LocalTask):
    """
    LabelBlockMapping on local machine
    """
    pass


class LabelBlockMappingSlurm(LabelBlockMappingBase, SlurmTask):
    """
    LabelBlockMapping on slurm cluster
    """
    pass


class LabelBlockMappingLSF(LabelBlockMappingBase, LSFTask):
    """
    LabelBlockMapping on lsf cluster
    """
    pass


#
# Implementation
#


def label_block_mapping(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    if 'number_of_labels' in config:
        number_of_labels = config['number_of_labels']

    elif 'unique_path' in config:
        unique_path = config['unique_path']
        unique_key = config['unique_key']
        with open_file(unique_path, 'r') as f:
            labels = f[unique_key][:]
        number_of_labels = None

    roi_begin = config.get('roi_begin', None)
    roi_end = config.get('roi_end', None)
    assert (roi_begin is None) == (roi_end is None)

    # we need to turn `None` rois to empty lists,
    # because I don't really understand how pybind11 handles None yet
    if roi_begin is None:
        roi_begin = []
        roi_end = []

    n_threads = config.get('threads_per_job', 1)
    if number_of_labels is None:
        pass
    else:
        ndist.serializeBlockMapping(input_path, input_key,
                                    output_path, output_key,
                                    number_of_labels, n_threads,
                                    roi_begin, roi_end)
    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    label_block_mapping(job_id, path)
