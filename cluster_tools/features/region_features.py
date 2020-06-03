#! /usr/bin/python

import os
import sys
import json

from elf.util import set_numpy_threads
set_numpy_threads(1)
import numpy as np

import luigi
import z5py
import nifty.tools as nt
import vigra

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class RegionFeaturesBase(luigi.Task):
    """ Block edge feature base class
    """

    task_name = 'region_features'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    channel = luigi.IntParameter(default=None)
    prefix = luigi.Parameter(default='')
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    # specify features once we support more than mean here
    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'ignore_label': 0})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # get shape and check dimension and channel param
        shape = vu.get_shape(self.input_path, self.input_key)
        if len(shape) == 4 and self.channel is None:
            raise RuntimeError("Got 4d input, but channel was not specified")
        if len(shape) == 4 and self.channel >= shape[0]:
            raise RuntimeError("Channel %i is to large for n-channels %i" % (self.channel,
                                                                             shape[0]))
        if len(shape) == 3 and self.channel is not None:
            raise RuntimeError("Channel was specified, but input is only 3d")

        if len(shape) == 4:
            shape = shape[1:]

        # temporary output dataset
        output_path = os.path.join(self.tmp_folder, 'region_features_tmp.n5')
        output_key = 'block_feats'

        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'labels_path': self.labels_path, 'labels_key': self.labels_key,
                       'output_path': output_path, 'output_key': output_key,
                       'block_shape': block_shape, 'channel': self.channel})

        # require the temporary output data-set
        f_out = z5py.File(output_path)
        f_out.require_dataset(output_key, shape=shape, compression='gzip',
                              chunks=tuple(block_shape), dtype='float32')

        if self.n_retries == 0:
            # get shape and make block config
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
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


class RegionFeaturesLocal(RegionFeaturesBase, LocalTask):
    """ RegionFeatures on local machine
    """
    pass


class RegionFeaturesSlurm(RegionFeaturesBase, SlurmTask):
    """ RegionFeatures on slurm cluster
    """
    pass


class RegionFeaturesLSF(RegionFeaturesBase, LSFTask):
    """ RegionFeatures on lsf cluster
    """
    pass


#
# Implementation
#


# we need to implement relabel sequential ourselves,
# because vigra.analysis.relabelConsecutive messes with the order
# and skimage.segmentation.relabel_sequential is too memory hungry
def relabel_sequential(data, unique_values):
    start_val = 0 if unique_values[0] == 0 else 1
    relabeling = {val: ii for ii, val in enumerate(unique_values, start_val)}
    return nt.takeDict(relabeling, data)


def _block_features(block_id, blocking,
                    ds_in, ds_labels, ds_out,
                    ignore_label, channel,
                    feature_names):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    labels = ds_labels[bb]

    # check if we have an ignore label and return
    # if this block is purely ignore label
    if ignore_label is not None:
        if np.sum(labels != ignore_label) == 0:
            fu.log_block_success(block_id)
            return

    # get global normalization values
    min_val = 0
    max_val = 255. if ds_in.dtype == np.dtype('uint8') else 1.

    bb_in = bb if channel is None else (channel,) + bb
    input_ = ds_in[bb_in]
    input_ = vu.normalize(input_, min_val, max_val)

    ids = np.unique(labels)
    if ids[0] == 0:
        feat_slice = np.s_[:]
        exp_len = len(ids)
    else:
        feat_slice = np.s_[1:]
        exp_len = len(ids) + 1

    # relabel consecutive in order to save memory
    labels = relabel_sequential(labels, ids)

    feats = vigra.analysis.extractRegionFeatures(input_, labels.astype('uint32'), features=feature_names,
                                                 ignoreLabel=ignore_label)
    assert len(feats['count']) == exp_len, "%i, %i" % (len(feats['count']), exp_len)

    # make serialization
    n_cols = len(feature_names) + 1
    data = np.zeros(n_cols * len(ids), dtype='float32')
    # write the ids
    data[::n_cols] = ids.astype('float32')
    # write all the features
    for feat_id, feat_name in enumerate(feature_names, 1):
        data[feat_id::n_cols] = feats[feat_name][feat_slice]

    chunks = blocking.blockShape
    chunk_id = tuple(b.start // ch for b, ch in zip(bb, chunks))
    ds_out.write_chunk(chunk_id, data, True)
    fu.log_block_success(block_id)


def region_features(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    block_list = config['block_list']
    input_path = config['input_path']
    input_key = config['input_key']
    labels_path = config['labels_path']
    labels_key = config['labels_key']
    output_path = config['output_path']
    output_key = config['output_key']
    block_shape = config['block_shape']
    channel = config['channel']
    ignore_label = config['ignore_label']

    # TODO there are some issues with min and max I don't understand
    # feature_names = ['count', 'mean', 'minimum', 'maximum']
    feature_names = ['count', 'mean']

    with vu.file_reader(input_path, 'r') as f_in,\
            vu.file_reader(labels_path, 'r') as f_l,\
            vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_labels = f_l[labels_key]
        ds_out = f_out[output_key]

        shape = ds_out.shape
        blocking = nt.blocking([0, 0, 0], shape, block_shape)

        for block_id in block_list:
            _block_features(block_id, blocking,
                            ds_in, ds_labels, ds_out,
                            ignore_label, channel,
                            feature_names)

        # write the feature names in job 0
        if job_id == 0:
            ds_out.attrs['feature_names'] = feature_names

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    region_features(job_id, path)
