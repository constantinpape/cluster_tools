#! /bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# copy_to_h5 tasks
#

class CopyToH5Base(luigi.Task):
    """ copy_to_h5 base class
    """

    task_name = 'copy_to_h5'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    effective_scale_factor = luigi.Parameter()
    prefix = luigi.Parameter()
    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'chunks': None, 'compression': 'gzip', 'dtype': None})
        return config

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape
            ds_dtype = f[self.input_key].dtype
            ds_chunks = f[self.input_key].chunks
        assert len(shape) == 3, "Only support 3d inputs"

        # load the copy_to_h5 config
        task_config = self.get_task_config()
        compression = task_config.pop('compression', 'gzip')

        dtype = task_config.pop('dtype', None)
        if dtype is None:
            dtype = ds_dtype

        if isinstance(dtype, np.dtype):
            dtype = dtype.name

        chunks = task_config.pop('chunks', None)
        if chunks is None:
            chunks = ds_chunks
        chunks = tuple(chunks)

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key,
                            'block_shape': block_shape, 'dtype': dtype})

        if roi_begin is not None:

            # if we have a roi, we need to re-sample it
            assert roi_end is not None
            effective_scale = self.effective_scale_factor if self.effective_scale_factor else scale_factor
            if isinstance(effective_scale, int):
                roi_begin = [rb // effective_scale for rb in roi_begin]
                roi_end= [re // effective_scale if re is not None else sh
                          for re, sh in zip(roi_end, shape)]
            else:
                roi_begin = [rb // sf for rb, sf in zip(roi_begin, effective_scale)]
                roi_end = [re // sf if re is not None else sh
                           for re, sf, sh in zip(roi_end, effective_scale, shape)]

            shape = tuple(roie - roib for roib, roie in zip(roi_begin, roi_end))
            task_config.update({'roi_begin': roi_begin, 'roi_end': roi_end})

        # check that the chunks are smaller than the shape, otherwise
        # skip this job
        if any(ch > sh for ch, sh in zip(chunks, shape)):
            self._write_log("chunks %s are bigger than shape %s, not scheduling any jobs" % (str(chunks), str(shape)))
            return

        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression=compression, dtype=dtype)


        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
            self._write_log("scheduled %i blocks to run" % len(block_list))
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config, self.prefix)
        self.submit_jobs(n_jobs, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.prefix)
        self.check_jobs(n_jobs, self.prefix)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.prefix))


class CopyToH5Local(CopyToH5Base, LocalTask):
    """
    copy to h5 on local machine
    """
    pass


class CopyToH5Slurm(CopyToH5Base, SlurmTask):
    """
    copy to h5 on slurm cluster
    """
    pass


class CopyToH5LSF(CopyToH5Base, LSFTask):
    """
    copy to h5 on lsf cluster
    """
    pass


#
# Implementation
#


def _copy_blocks(ds_in, ds_out, block_shape, block_list,
                 dtype, roi_begin=None, roi_end=None):

    if roi_begin is None:
        blocking = nt.blocking([0, 0, 0], list(ds_in.shape), list(block_shape))
    else:
        blocking = nt.blocking(roi_begin, roi_end, list(block_shape))

    for block_id in range(blocking.numberOfBlocks):
        fu.log("start processing block %i" % block_id)
        block = blocking.getBlock(block_id)
        bb_in = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        if roi_begin is not None:
            bb_out = tuple(slice(beg - off, end - off)
                           for beg, end, off in zip(block.begin, block.end, roi_begin))
        else:
            bb_out = bb_in
        data = ds_in[bb_in]
        # don't write empty blocks
        if np.sum(data) == 0:
            fu.log_block_success(block_id)
            continue
        ds_out[bb_out] = data.astype(dtype)
        fu.log_block_success(block_id)


def copy_to_h5(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    # read the output config
    output_path = config['output_path']
    output_key = config['output_key']
    dtype = config['dtype']

    roi_begin = config.get('roi_begin', None)
    roi_end = config.get('roi_end', None)
    assert (roi_begin is None) == (roi_end is None)
    n_threads = config.get('threads_per_job', 1)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in  = f_in[input_key]
        ds_in.n_threads = n_threads
        ds_out = f_out[output_key]
        _copy_blocks(ds_in, ds_out, block_shape, block_list,
                     dtype, roi_begin, roi_end)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    copy_to_h5(job_id, path)
