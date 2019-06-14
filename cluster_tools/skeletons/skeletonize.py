#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt

import skeletor.io as skelio
from skeletor import skeletonize as skel_impl, get_method_names

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# skeletonize tasks
#


class SkeletonizeBase(luigi.Task):
    """ Skeletonize base class
    """

    task_name = 'skeletonize'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    morphology_path = luigi.Parameter()
    morphology_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    number_of_labels = luigi.IntParameter()
    resolution = luigi.ListParameter()
    size_threshold = luigi.IntParameter(default=None)
    method = luigi.Parameter(default='thinning')
    dependency = luigi.TaskParameter(default=DummyTask())

    methods = get_method_names()

    # TODO expose skeletonization parameter
    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'chunk_len': 1000,
                       'method_kwargs': {}})
        return config

    def requires(self):
        return self.dependency

    def _prepare_output(self, config):
        # make the blocking
        block_len = min(self.number_of_labels, config.get('chunk_len', 1000))
        block_list = vu.blocks_in_volume((self.number_of_labels,),
                                         (block_len,))
        n_jobs = min(len(block_list), self.max_jobs)
        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=(self.number_of_labels,),
                              chunks=(1,), compression='gzip', dtype='uint64')
        # update the config
        config.update({'number_of_labels': self.number_of_labels,
                       'block_len': block_len})
        return config, n_jobs, block_list

    def run_impl(self):
        assert self.method in self.methods,\
            "Method %s is not supported, must be one of %s" % (self.method, str(self.methods))

        # TODO support roi
        # get the global config and init configs
        shebang, block_shape, _, _ = self.global_config_values()
        self.init(shebang)

        # load the skeletonize config
        # update the config with input and output paths and keys
        config = self.get_task_config()
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'morphology_path': self.morphology_path, 'morphology_key': self.morphology_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'resolution': self.resolution, 'size_threshold': self.size_threshold,
                       'method': self.method})
        config, n_jobs, block_list = self._prepare_output(config)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class SkeletonizeLocal(SkeletonizeBase, LocalTask):
    """
    skeletonize on local machine
    """
    pass


class SkeletonizeSlurm(SkeletonizeBase, SlurmTask):
    """
    skeletonize on slurm cluster
    """
    pass


class SkeletonizeLSF(SkeletonizeBase, LSFTask):
    """
    skeletonize on lsf cluster
    """
    pass


#
# Implementation
#


# not parallelized for now
def _skeletonize_id_block(blocking, block_id, ds_in, ds_out,
                          sizes, bb_min, bb_max, resolution, size_threshold,
                          method):

    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    id_begin, id_end = block.begin[0], block.end[0]
    # we don't compute the skeleton for id 0, which is reserved for the ignore label
    id_begin = 1 if id_begin == 0 else id_begin

    # skeletonize ids in range and serialize skeletons
    for seg_id in range(id_begin, id_end):
        if size_threshold is not None:
            if sizes[seg_id] < size_threshold:
                continue
        bb = tuple(slice(mi, ma) for mi, ma in zip(bb_min[seg_id], bb_max[seg_id]))
        obj = ds_in[bb] == seg_id
        nodes, edges = skel_impl(obj, resolution=resolution, method=method)

        offsets = [b.start for b in bb]
        skelio.write_n5(ds_out, seg_id,
                        nodes, edges, offsets)
    fu.log_block_success(block_id)


def skeletonize(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']
    morphology_path = config['morphology_path']
    morphology_key = config['morphology_key']
    output_path = config['output_path']
    output_key = config['output_key']
    size_threshold = config['size_threshold']
    resolution = config['resolution']
    method = config['method']

    # morphology feature-columns
    # 0    = label-id
    # 1    = pixel size
    # 2:5  = center of mass
    # 5:8  = min coordinate
    # 8:11  = max coordinate
    with vu.file_reader(morphology_path) as f:
        morpho = f[morphology_key][:]
        sizes = morpho[:, 1].astype('uint64')
        bb_min = morpho[:, 5:8].astype('uint64')
        bb_max = morpho[:, 8:11].astype('uint64') + 1

    block_list = config['block_list']
    block_len = config['block_len']
    n_labels = config['number_of_labels']
    blocking = nt.blocking([0], [n_labels], [block_len])

    # skeletonize this id block
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]
        for block_id in block_list:
            _skeletonize_id_block(blocking, block_id, ds_in, ds_out,
                                  sizes, bb_min, bb_max, resolution, size_threshold,
                                  method)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    skeletonize(job_id, path)
