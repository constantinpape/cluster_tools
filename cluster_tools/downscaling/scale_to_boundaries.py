#! /bin/python

# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import os
import sys
import json

import numpy as np

import luigi
import nifty.tools as nt
from elf.wrapper.resized_volume import ResizedVolume

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# scaling tasks
#


class ScaleToBoundariesBase(luigi.Task):
    """ scale_to_boundaries base class
    """

    task_name = 'scale_to_boundaries'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    boundaries_path = luigi.Parameter()
    boundaries_key = luigi.Parameter()
    offset = luigi.IntParameter(default=0)
    dependency = luigi.TaskParameter(default=DummyTask())

    allow_retry = False

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        # Parameters:
        # erode_by - erosion of seeds.
        # channel -  channel that will be used for multiscale inputs
        # dtype - output dtype
        # chunks - output chunks
        config.update({'erode_by': 12, 'channel': 0, 'dtype': 'uint64', 'chunks': None,
                       'erode_3d': True})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        shape = vu.get_shape(self.boundaries_path, self.boundaries_key)
        if len(shape) == 4:
            shape = shape[1:]
        assert len(shape) == 3

        # require output dataset
        config = self.get_task_config()
        dtype = config.pop('dtype')
        chunks = config.pop('chunks')
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)
        assert all(bs % ch == 0 for bs, ch in zip(block_shape, chunks)),\
            "%s, %s" % (str(block_shape), str(chunks))
        self._write_log("requiring output dataset @ %s:%s" % (self.output_path, self.output_key))
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=tuple(chunks),
                              compression='gzip', dtype=dtype)

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'boundaries_path': self.boundaries_path, 'boundaries_key': self.boundaries_key,
                       'offset': self.offset, 'block_shape': block_shape})

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        self._write_log("scheduled %i blocks to run" % len(block_list))

        prefix = 'offset%i' % self.offset
        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, config, prefix)
        self.submit_jobs(n_jobs, prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(prefix)
        self.check_jobs(n_jobs, prefix)

    def output(self):
        prefix = 'offset%i' % self.offset
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % prefix))


class ScaleToBoundariesLocal(ScaleToBoundariesBase, LocalTask):
    """
    scale_to_boundaries on local machine
    """
    pass


class ScaleToBoundariesSlurm(ScaleToBoundariesBase, SlurmTask):
    """
    scale_to_boundaries on slurm cluster
    """
    pass


class ScaleToBoundariesLSF(ScaleToBoundariesBase, LSFTask):
    """
    scale_to_boundaries on lsf cluster
    """
    pass


#
# Implementation
#


@threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
def compute_halo(erode_by, erode_3d):
    if isinstance(erode_by, int):
        halo = erode_by
    else:
        assert isinstance(erode_by, dict), 'Need int or dict'
        halo = max(erode_by.values())
    halo = 3 * [halo] if erode_3d else [0, halo, halo]
    return halo


@threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
def _scale_block(block_id, blocking,
                 ds_in, ds_bd, ds_out,
                 offset, erode_by, erode_3d, channel):
    fu.log("start processing block %i" % block_id)
    # load the block with halo set to 'erode_by'
    halo = compute_halo(erode_by, erode_3d)
    block = blocking.getBlockWithHalo(block_id, halo)
    in_bb = vu.block_to_bb(block.outerBlock)
    out_bb = vu.block_to_bb(block.innerBlock)
    local_bb = vu.block_to_bb(block.innerBlockLocal)

    obj = ds_in[in_bb]
    # don't scale if block is empty
    if np.sum(obj != 0) == 0:
        fu.log_block_success(block_id)
        return

    # load boundary map and fit obj to it
    if ds_bd.ndim == 4:
        in_bb = (slice(channel, channel + 1),) + in_bb
    hmap = ds_bd[in_bb].squeeze()
    obj, _ = vu.fit_to_hmap(obj, hmap, erode_by, erode_3d)
    obj = obj[local_bb]

    fg_mask = obj != 0
    obj[fg_mask] += offset

    # load previous output volume, insert obj into it and save again
    out = ds_out[out_bb]
    out[fg_mask] += obj[fg_mask]
    ds_out[out_bb] = out
    # log block success
    fu.log_block_success(block_id)


def scale_to_boundaries(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read paths from the config
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    boundaries_path = config['boundaries_path']
    boundaries_key = config['boundaries_key']
    offset = config['offset']

    # additional config
    erode_by = config['erode_by']
    erode_3d = config.get('erode_3d', True)
    channel = config['channel']

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    with vu.file_reader(input_path, 'r') as fin,\
            vu.file_reader(boundaries_path, 'r') as fb,\
            vu.file_reader(output_path) as fout:

        ds_bd = fb[boundaries_key]
        ds_out = fout[output_key]

        shape = ds_out.shape
        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)

        ds_in = ResizedVolume(fin[input_key], shape)

        for block_id in block_list:
            _scale_block(block_id, blocking,
                         ds_in, ds_bd, ds_out,
                         offset, erode_by, erode_3d, channel)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    scale_to_boundaries(job_id, path)
