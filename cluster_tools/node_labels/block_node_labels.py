#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt
import nifty.distributed as ndist
from elf.wrapper.resized_volume import ResizedVolume
from elf.label_multiset import deserialize_labels

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

    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    ignore_label = luigi.IntParameter(default=None)
    prefix = luigi.Parameter(default='')
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'ws_path': self.ws_path, 'ws_key': self.ws_key,
                       'input_path': self.input_path,
                       'input_key': self.input_key,
                       'block_shape': block_shape,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'ignore_label': self.ignore_label})

        with vu.file_reader(self.ws_path, 'r') as f:
            ds = f[self.ws_key]
            shape = ds.shape

            chunks = tuple(min(bs, sh) for bs, sh in zip(block_shape, shape))
            attrs = ds.attrs

            try:
                max_id = attrs['maxId']
            except KeyError:
                raise KeyError("Dataset %s:%s does not have attribute maxId" % (self.ws_path,
                                                                                self.ws_key))
            is_label_multiset = attrs.get('isLabelMultiset', False)

            if is_label_multiset:
                assert all(ch == dch for ch, dch in zip(chunks, ds.chunks)), "%s, %s" % (str(chunks), str(ds.chunks))

        # create output dataset
        with vu.file_reader(self.output_path) as f:
            ds_out = f.require_dataset(self.output_key, shape=shape,
                                       dtype='uint64',
                                       chunks=chunks,
                                       compression='gzip')
            # need to serialize the label max-id here for
            # the merge_node_labels task
            ds_out.attrs['maxId'] = int(max_id)

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape,
                                             roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config, self.prefix)
        self.submit_jobs(n_jobs, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs, self.prefix)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.prefix))


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

def _load_block(ds, bb, is_label_multiset):
    if is_label_multiset:
        chunks = ds.chunks
        start = [b.start for b in bb]
        chunk_id = tuple(st // ch for st, ch in zip(start, chunks))
        bb_shape = tuple(b.stop - b.start for b in bb)
        data = ds.read_chunk(chunk_id)
        data = np.zeros(bb_shape, dtype='uint32') if data is None else\
            deserialize_labels(data, bb_shape)
    else:
        data = ds[bb]
    return data


def _labels_for_block(block_id, blocking,
                      ds_ws, output_path, output_key,
                      labels, ignore_label, is_label_multiset):
    fu.log("start processing block %i" % block_id)
    # read labels and input in this block
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)
    ws = _load_block(ds_ws, bb, is_label_multiset)

    # check if watershed block is empty
    if ws.sum() == 0:
        fu.log("block %i is empty" % block_id)
        fu.log_block_success(block_id)
        return

    # serialize the overlaps
    labs = labels[bb].astype('uint64')

    # check if label block is empty:
    if ignore_label is not None:
        if np.sum(labs == ignore_label) == labs.size:
            fu.log("labels of block %i is empty" % block_id)
            fu.log_block_success(block_id)
            return

    chunk_id = tuple(beg // ch
                     for beg, ch in zip(block.begin,
                                        blocking.blockShape))
    ndist.computeAndSerializeLabelOverlaps(ws, labs,
                                           output_path, output_key, chunk_id,
                                           withIgnoreLabel=False if ignore_label is None else True,
                                           ignoreLabel=0 if ignore_label is None else ignore_label)
    fu.log_block_success(block_id)


def block_node_labels(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    ws_path = config['ws_path']
    ws_key = config['ws_key']
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']

    block_shape = config['block_shape']
    block_list = config['block_list']
    ignore_label = config['ignore_label']

    with vu.file_reader(ws_path, 'r') as f:
        shape = f[ws_key].shape

    blocking = nt.blocking([0, 0, 0],
                           list(shape),
                           list(block_shape))

    # labels can either be interpolated or full volume
    f_lab = vu.file_reader(input_path, 'r')
    ds_labels = f_lab[input_key]
    lab_shape = ds_labels.shape
    # label shape is smaller than ws shape
    # -> interpolated
    if any(lsh < sh for lsh, sh in zip(lab_shape, shape)):
        assert not any(lsh > sh for lsh, sh in zip(lab_shape, shape)),\
            "Can't have label shape bigger then volshape"
        labels = ResizedVolume(ds_labels, shape, order=0)
    else:
        assert lab_shape == shape, "%s, %s" % (str(lab_shape), shape)
        labels = ds_labels

    if ignore_label is None:
        fu.log("accumulating labels without ignore label")
    else:
        fu.log("accumulating labels with ignore label %i" % ignore_label)

    with vu.file_reader(ws_path, 'r') as f_in:
        ds_ws = f_in[ws_key]
        is_label_multiset = ds_ws.attrs.get('isLabelMultiset', False)
        [_labels_for_block(block_id, blocking,
                           ds_ws, output_path, output_key,
                           labels, ignore_label, is_label_multiset)
         for block_id in block_list]

    f_lab.close()
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    block_node_labels(job_id, path)
