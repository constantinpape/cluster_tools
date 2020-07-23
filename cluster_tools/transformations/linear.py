#! /bin/python

import os
import sys
import json

import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# linear transformation tasks
#

class LinearBase(luigi.Task):
    """ linear base class
    """

    task_name = 'linear'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    transformation = luigi.Parameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')
    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'chunks': None, 'compression': 'gzip'})
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
            dtype = f[self.input_key].dtype

        # load the config
        task_config = self.get_task_config()
        compression = task_config.pop('compression', 'gzip')
        chunks = task_config.pop('chunks', None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)

        if self.output_path != self.input_path:
            # require output dataset
            with vu.file_reader(self.output_path) as f:
                f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                                  compression=compression, dtype=dtype)

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'output_path': self.output_path, 'output_key': self.output_key,
                            'mask_path': self.mask_path, 'mask_key': self.mask_key,
                            'block_shape': block_shape, 'transformation': self.transformation})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        self._write_log("scheduled %i blocks to run" % len(block_list))

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class LinearLocal(LinearBase, LocalTask):
    """
    Linear intensity transform on local machine
    """
    pass


class LinearSlurm(LinearBase, SlurmTask):
    """
    copy on slurm cluster
    Linear intensity transform on slurm cluster
    """
    pass


class LinearLSF(LinearBase, LSFTask):
    """
    Linear intensity transform on lsf cluster
    """
    pass


#
# Implementation
#


def _load_transformation(trafo_file, shape):
    with open(trafo_file) as f:
        trafo = json.load(f)

    # for now, we support two different transformation specifications:
    # 1.) global trafo specified as {'a': a, 'b': b}
    # 2.) transformation for each slcie, specified as {'1': {'a': a, 'b': b}, ...}
    if len(trafo) == 2:
        assert set(trafo.keys()) == {'a', 'b'}
        fu.log("Found global transformation with values %f, %f" % (trafo['a'], trafo['b']))
    else:
        assert len(trafo) == shape[0]
        assert all((len(tr) == 2 for tr in trafo.values()))
        trafo = {int(k): v for k, v in trafo.items()}
        fu.log("Found transformation per slice")
    return trafo


def _transform_data(data, a, b, mask=None):
    if mask is None:
        data = a * data + b
    else:
        data[mask] = a * data[mask] + b
    return data


def _transform_block(ds_in, ds_out, transformation, blocking, block_id, mask=None):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)

    bb = vu.block_to_bb(block)
    if mask is not None:
        bb_mask = mask[bb].astype('bool')
        if bb_mask.sum() == 0:
            fu.log_block_success(block_id)
            return
    else:
        bb_mask = None

    data = ds_in[bb]
    if len(transformation) == 2:
        data = _transform_data(data, transformation['a'], transformation['b'], bb_mask)
    else:
        z_offset = block.begin[0]
        for z in range(data.shape[0]):
            trafo = transformation[z + z_offset]
            data[z] = _transform_data(data[z], trafo['a'], trafo['b'], bb_mask[z])

    ds_out[bb] = data
    fu.log_block_success(block_id)


def _transform_linear(ds_in, ds_out, transformation, blocking, block_list, mask=None):
    for block_id in block_list:
        _transform_block(ds_in, ds_out, transformation, blocking, block_id, mask)


def linear(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    # read the output config and path to transformation
    output_path = config['output_path']
    output_key = config['output_key']
    trafo_file = config['transformation']

    mask_path = config['mask_path']
    mask_key = config['mask_key']

    if mask_path != '':
        assert mask_key != ''
        with vu.file_reader(input_path, 'r') as f:
            in_shape = f[input_key].shape
        mask = vu.load_mask(mask_path, mask_key, in_shape)

    same_file = input_path == output_path
    in_place = same_file and (input_key == output_key)

    # submit blocks
    if same_file:
        with vu.file_reader(input_path) as f:
            ds_in = f[input_key]
            ds_out = ds_in if in_place else f[output_key]

            shape = list(ds_in.shape)
            trafo = _load_transformation(trafo_file, shape)

            blocking = nt.blocking([0, 0, 0], shape, block_shape)
            _transform_linear(ds_in, ds_out, trafo, blocking, block_list, mask)

    else:
        with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
            ds_in = f_in[input_key]
            ds_out = f_out[output_key]

            shape = list(ds_in.shape)
            trafo = _load_transformation(trafo_file, shape)

            blocking = nt.blocking([0, 0, 0], shape, block_shape)
            _transform_linear(ds_in, ds_out, trafo, blocking, block_list, mask)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    linear(job_id, path)
