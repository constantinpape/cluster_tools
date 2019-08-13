#! /bin/python

import os
import sys
import json

# this is a task called by multiple processes,
# so we need to restrict the number of threads used by numpy
# from cluster_tools.utils.numpy_utils import set_numpy_threads
# set_numpy_threads(1)
# import numpy as np

import luigi
import nifty.tools as nt
# import vigra

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class LabelMultisetBase(luigi.Task):
    """ LabelMultiset base class
    """

    task_name = 'label_multiset'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    scale_factor = luigi.Parameter()
    restrict_set = luigi.Parameter()

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        config.update({'compression': 'gzip'})
        return config

    def downsample_shape(self, in_shape):
        scale_factor = self.scale_factor if isinstance(self.scale_factor, (tuple, list))\
            else [self.scale_factor] * len(in_shape)
        return tuple(sh // sc for sh, sc in zip(in_shape, scale_factor))

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        # load the label_multiset config
        config = self.get_task_config()

        compression = config.get('compression', 'gzip')
        out_shape = self.downsample_shape(shape)
        # require output dataset
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=out_shape, chunks=tuple(block_shape),
                              compression=compression, dtype='uint8')

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'scale_factor': self.scale_factor, 'restrict_set': self.restrict_set,
                       'block_shape': block_shape})
        block_list = vu.blocks_in_volume(out_shape, block_shape, roi_begin, roi_end)
        self._write_log('scheduling %i blocks to be processed' % len(block_list))
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class LabelMultisetLocal(LabelMultisetBase, LocalTask):
    """
    LabelMultiset on local machine
    """
    pass


class LabelMultisetSlurm(LabelMultisetBase, SlurmTask):
    """
    LabelMultiset on slurm cluster
    """
    pass


class LabelMultisetLSF(LabelMultisetBase, LSFTask):
    """
    LabelMultiset on lsf cluster
    """
    pass


#
# Implementation
#


# need to handle normal labels (scale 0) and multi-label-sets (scale > 0)
def load_input_data(ds_in, bb_in):
    try:
        data = ds_in[bb_in]
        is_label_multiset = True
    # TODO only check or the relevant exception
    except Exception:
        chunk_id = tuple(b.start // ch for b, ch in zip(bb_in, ds_in.chunks))
        data = ds_in.read_chunk(chunk_id)
        is_label_multiset = False
    return data, is_label_multiset


def compute_multiset_from_labels(labels, scale_factor, restrict_set):
    pass


def compute_multiset_from_multiset(multi_set, scale_factor, restrict_set):
    pass


def _label_multiset_block(blocking, block_id, ds_in, ds_out,
                          scale_factor, restrict_set):
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    bb_in = tuple(slice(b.start * scale, b.stop * scale)
                  for b, scale in zip(bb, scale_factor))

    data, is_label_multiset = load_input_data(ds_in, bb_in)
    # compute multiset from input data
    multi_set = compute_multiset_from_multiset(data,
                                               scale_factor,
                                               restrict_set) if is_label_multiset else\
        compute_multiset_from_labels(data,
                                     scale_factor,
                                     restrict_set)

    chunk_id = tuple(bs // ch for bs, ch in zip(block.begin, ds_out.chunks))
    ds_out.write_chunk(chunk_id, multi_set, True)


def _write_metadata(ds_out, max_id, restrict_set, scale_factor):
    attrs = ds_out.attrs
    attrs['maxId'] = max_id
    attrs['isLabelMultiset'] = True
    attrs['maxNumEntries'] = restrict_set
    attrs['downsamlingFactors'] = [float(sf) for sf in scale_factor]


def label_multiset(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']
    restrict_set = config['restrict_set']
    restrict_set = -1 if restrict_set is None else restrict_set
    scale_factor = config['scale_factor']
    scale_factor = scale_factor if isinstance(scale_factor, list) else [scale_factor] * 3

    block_shape = list(config['block_shape'])
    block_list = config['block_list']

    # read the output config
    output_path = config['output_path']
    output_key = config['output_key']
    shape = list(vu.get_shape(output_path, output_key))

    # get the blocking
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # submit blocks
    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        for block_id in block_list:
            _label_multiset_block(blocking, block_id, ds_in, ds_out,
                                  scale_factor, restrict_set)

        if job_id == 0:
            max_id = ds_in.attrs['maxId']
            _write_metadata(ds_out, max_id, restrict_set, scale_factor)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    label_multiset(job_id, path)