#! /bin/python

import os
import sys
import json
import functools

import luigi
import dask
import toolz as tz
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from .frameworks import get_predictor, get_preprocessor


#
# Inference Tasks
#


class InferenceBase(luigi.Task):
    """ Inference base class
    """

    task_name = 'inference'
    src_file = os.path.abspath(__file__)

    # input volumes and graph
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    checkpoint_path = luigi.Parameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')
    framework = luigi.Parameter(default='pytorch')
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'n_channels': 1, 'dtype': 'uint8', 'compression': 'compression',
                       'chunks': (25, 256, 256), 'halo': None})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run(self):
        assert self.framework in ('pytorch', 'tensorflow', 'caffe', 'inferno')

        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        n_channels = confg.get('n_channels', 1)
        dtype = config.pop('dtype', 'uint8')
        compression = config.pop('compression', 'gzip')
        chunks = tuple(config.pop('chunks', (25, 256, 256)))
        assert dtype in ('uint8', 'float32')

        # get shapes
        shape = vu.get_shape(self.input_path, self.input_key)
        if n_channels > 1:
            out_shape = (n_channels,) + shape
            chunks = (min(3, n_channels),) + chunks
        else:
            out_shape = shape

        # make output volume
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=out_shape,
                              chunks=out_chunks, dtype=dtype, compression=compression)

        # update the config
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape})

        if self.mask_path != '':
            assert self.mask_key != ''
            config.update({'mask_path': self.mask_path, 'mask_key': self.mask_key})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class InferenceLocal(InferenceBase, LocalTask):
    """ Inference on local machine
    """
    pass


class InferenceSlurm(InferenceBase, SlurmTask):
    """ Inference on slurm cluster
    """
    def _write_slurm_file(self, job_prefix=None):
        groupname = self.get_global_config().get('groupname', 'kreshuk')
        # read and parse the relevant task config
        task_config = self.get_task_config()
        n_threads = task_config.get("threads_per_job", 1)
        time_limit = self._parse_time_limit(task_config.get("time_limit", 60))
        mem_limit = self._parse_mem_limit(task_config.get("mem_limit", 2))

        # get file paths
        trgt_file = os.path.join(self.tmp_folder, self.task_name + '.py')
        config_tmpl = self._config_path('$1', job_prefix)
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name, job_prefix)
        # TODO set the job-name so that we can parse the squeue output properly
        slurm_template = ("#!/bin/bash\n"
                          "#SBATCH -A %s\n"
                          "#SBATCH -N 1\n"
                          "#SBATCH -n %i\n"
                          "#SBATCH --mem %s\n"
                          "#SBATCH -t %s\n"
                          '#SBATCH -p gpu\n'
                          '#SBATCH -C gpu=1080Ti'
                          '#SBATCH --gres=gpu:1'
                          "%s %s") % (groupname, n_threads,
                                      mem_limit, time_limit,
                                      trgt_file, config_tmpl)
        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % job_name)
        with open(script_path, 'w') as f:
            f.write(slurm_template)


class InferenceLSF(InferenceBase, LSFTask):
    """ Inference on lsf cluster
    """
    pass


#
# Implementation
#

# TODO figure out padding
def _load_input(ds, bb, block_shape, padding_mode='reflect'):

    actual_shape = tuple(b.stop - b.start for b in bb)

    # we pad the input volume if necessary
    pad_left = None
    pad_right = None

    # check for padding to the left
    if any(start < 0 for start in starts):
        pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
        starts = [max(0, start) for start in starts]

    # check for padding to the right
    if any(stop > shape[i] for i, stop in enumerate(stops)):
        pad_right = tuple(stop - shape[i] if stop > shape[i] else 0 for i, stop in enumerate(stops))
        stops = [min(shape[i], stop) for i, stop in enumerate(stops)]

    bb = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    data = io.read(bb)

    # pad if necessary
    if pad_left is not None or pad_right is not None:
        pad_left = (0, 0, 0) if pad_left is None else pad_left
        pad_right = (0, 0, 0) if pad_right is None else pad_right
        pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
        data = np.pad(data, pad_width, mode=padding_mode)

    return data


# TODO
def _to_uint8(data):
    pass


def _run_inference_with_mask(blocking, block_list):
    pass


def _run_inference(blocking, block_list, halo, ds_in, ds_out,
                   preprocess, predict, dtype, n_channels,
                   n_threasds):

    block_shape = blocking.blockShape

    @dask.delayed
    def log1(block_id):
        fu.log("start processing block %i" % block_id)
        return block_id

    @dask.delayed
    def load_input(block_id):
        if halo is None:
            block = blocking.getBlock(block_id)
        else:
            block = blocking.getBlockWithHalo(block_id, halo).outerBlock
        bb = vu.block_to_bb(block)
        return _load_input(ds_in, bb, block_shape)

    preprocess = dask.delayed(preprocess)
    predict = dask.delayed(predict)

    @dask.delayed
    def write_output(output):
        out_shape = output.shape
        if len(out_shape) == 4:
            assert out_shape[1:] == block_shape
            assert out_shape[0] >= n_channels
        else:
            assert out_shape == block_shape

        bb = vu.block_to_bb(blocking.getBlock(block_id))

        # adjust bounding box to multi-channel output
        if output.ndim == 4:
            output = output[:n_channels]
            bb = (slice(0, n_channels),) + bb

        # check if we need to crop the output
        actual_shape = tuple(b.stop - b.start for b in bb)
        if actual_shape != block_shape:
            block_bb = tuple(slice(0, bsh - ash) for bsh, ash in zip(block_shape, actual_shape))
            if output.ndim == 4:
                block_bb = (slice(None),) + block_bb
            output = output[block_bb]

        # cast to uint8 if necessary
        if dtype == 'uint8':
            output = _to_uint8(output)

        ds_out[bb] = output
        return block_id

    @dask.delayed
    def log2(block_id):
        fu.log_block_success(block_id)
        return 1

    # TODO need to pipe block-id
    # TODO can we pipe two values ?
    # iterate over the blocks in block list, get the input data and predict
    results = []
    for block_id in block_list:
        res = tz.pipe(block_id, log1, load_input, preprocess, predict, write_output, log2)
        results.append(res)

    get = functools.partial(dask.threaded.get, num_workers=num_cpus)
    # NOTE: Because dask.compute doesn't take an argument, but rather an
    # arbitrary number of arguments, computing each in turn, the output of
    # dask.compute(results) is a tuple of length 1, with its only element
    # being the results list. If instead we pass the results list as *args,
    # we get the desired container of results at the end.
    success = dask.compute(*results, get=get)
    fu.log('Finished prediction for %i blocks' % sum(success))


def inference(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    block_shape = config['block_shape']
    block_list = config['block_list']

    dtype = config.get('dtype', 'uint8')
    n_channels = config.get('n_channels', 1)
    halo = config.get('halo', None)
    framework = config.get('framework', 'pytorch')
    n_threads = config['n_threads']

    predict = get_predictor(framework)
    preprocess = get_preprocessor(framework)

    shape = vu.get_shape(input_path, input_key)
    blocking = nt.blocking(roiBegin=[0, 0, 0],
                           roiEnd=list(shape),
                           blockShape=list(block_shape))

    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_out = f_in[outputt_key]

        if mask_path in config:
            mask = vu.load_mask(config['mask_path'], config['mask_key'], shape)
            _run_inference_with_mask(blocking, block_list, halo, ds_in, ds_out, mask,
                                     predict, dtype, n_channels)
        else:
            _run_inference(blocking, block_list, halo, ds_in, ds_out,
                           preprocess, predict, dtype, n_channels, n_threads)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    inference(job_id, path)
