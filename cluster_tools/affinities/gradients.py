#! /usr/bin/python

import os
import sys
import json
import pickle

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Block-wise gradient computation tasks
#

class GradientsBase(luigi.Task):
    """ Gradients base class
    """

    task_name = 'gradients'
    src_file = os.path.abspath(__file__)
    allow_retry = True

    path_dict = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    average_gradient = luigi.BoolParameter(default=True)
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    def _validate_paths(self):
        shape = None

        with open(self.path_dict) as f:
            path_dict = json.load(f)

        for path in sorted(path_dict):
            key = path_dict[path]
            assert os.path.exists(path)
            with vu.file_reader(path, 'r') as f:
                assert key in f
                ds = f[key]
                if shape is None:
                    shape = ds.shape
                else:
                    # TODO support multi-channel inputs and then only check that
                    # spatial shapes agree
                    assert ds.shape == shape
        return shape

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)
        shape = self._validate_paths()

        config = self.get_task_config()
        config.update({'path_dict': self.path_dict,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'block_shape': block_shape,
                       'average_gradient': self.average_gradient})

        # TODO need to adapt to multi-channel
        chunks = tuple(min(bs // 2, sh) for bs, sh in zip(block_shape, shape))

        if self.average_gradient:
            out_shape = shape
            out_chunks = chunks
        else:
            n_channels = len(path_dict)
            out_shape = (n_channels,) + shape
            out_chunks = (1,) + chunks

        # make output dataset
        compression = config.pop('compression', 'gzip')
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=out_shape, dtype='float32',
                              compression=compression, chunks=out_chunks)

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class GradientsLocal(GradientsBase, LocalTask):
    """
    Gradients on local machine
    """
    pass


class GradientsSlurm(GradientsBase, SlurmTask):
    """
    Gradients on slurm cluster
    """
    pass


class GradientsLSF(GradientsBase, LSFTask):
    """
    Gradients on lsf cluster
    """
    pass


#
# Implementation
#


# np.gradient returns a list of len 3 corresponding to the gradient
# along each direction. I am not sure if we want to average
# along that as done here, or keep the dimension information
def _compute_average_gradients(input_datasets, shape, outer_bb):
    out_data = np.zeros(shape, dtype='float32')
    for chan, inds in enumerate(input_datasets):
        x = inds[outer_bb]
        x = np.array(np.gradient(x))
        x = np.mean(x, axis=0)
        assert x.shape == out_data.shape
        out_data = (x + out_data * chan) / (chan + 1)
    return out_data


def _compute_all_gradients(input_datasets, shape, outer_bb):
    out_data = np.zeros(shape, dtype='float32')
    for chan, inds in enumerate(input_datasets):
        x = inds[outer_bb]
        x = np.array(np.gradient(data))
        x = np.mean(x, axis=0)
        out_data[chan] = x
    return out_data


def _gradients_block(block_id, blocking,
                     input_datasets, ds, halo,
                     average_gradient):
    fu.log("start processing block %i" % block_id)

    block = blocking.getBlockWithHalo(block_id, halo)
    outer_bb = vu.block_to_bb(block.outerBlock)
    inner_bb = vu.block_to_bb(block.innerBlock)
    local_bb = vu.block_to_bb(block.innerBlockLocal)

    bshape = tuple(ob.stop - ob.start for ob in outer_bb)
    if average_gradient:
        out_shape = bshape
        out_data = _compute_average_gradients(input_datasets, out_shape, outer_bb)
    else:
        n_channels = len(input_datasets)
        out_shape = (n_channels,) + bshape
        out_data = _compute_all_gradients(input_datasets, out_shape, outer_bb)

        inner_bb = (slice(None),) + inner_bb
        local_bb = (slice(None),) + local_bb

    ds[inner_bb] = out_data[local_bb]
    fu.log_block_success(block_id)


def gradients(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    path_dict = config['path_dict']
    output_path = config['output_path']
    output_key = config['output_key']
    block_list = config['block_list']
    block_shape = config['block_shape']
    average_gradient = config['average_gradient']

    with open(path_dict) as f:
        path_dict = json.load(f)

    input_datasets = []
    for path in sorted(path_dict):
        input_datasets.append(vu.file_reader(path, 'r')[path_dict[path]])

    # 5 pix should be enough halo to make gradient computation correct
    halo = 3 * [5]
    with vu.file_reader(output_path) as f:
        ds = f[output_key]
        shape = ds.shape if average_gradient else ds.shape[1:]
        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)
        [_gradients_block(block_id, blocking, input_datasets,
                          ds, halo, average_gradient)
         for block_id in block_list]
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    gradients(job_id, path)
