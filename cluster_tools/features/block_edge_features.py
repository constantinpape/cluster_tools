#! /usr/bin/python

import os
import z5py
import argparse
import time
import json

import numpy as np
import nifty.distributed as ndist

from ..cluster_tasks import LocalTask, LSFTask, SlurmTask


# TODO implement watershed with mask
class BlockEdgeFeaturesBase(luigi.Task):
    """ Block edge feature base class
    """

    task_name = 'block_edge_features'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update()
        return config

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)
        if len(shape) == 4:
            shape = shape[1:]

        # load the watershed config
        ws_config = self.get_task_config()

        # require output dataset
        # TODO read chunks from config
        chunks = tuple(bs // 2 for bs in block_shape)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks,
                              compression='gzip', dtype='uint64')

        # update the config with input and output paths and keys
        # as well as block shape
        ws_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                          'output_path': self.output_path, 'output_key': self.output_key,
                          'block_shape': block_shape})

        # check if we run a 2-pass watershed
        is_2pass = ws_config.pop('two_pass', False)

        self._write_log("run one pass watershed")
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)
        self._watershed_pass(n_jobs, block_list, ws_config)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, ws_config, prefix)
        self.submit_jobs(n_jobs, prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(prefix)
        self.check_jobs(n_jobs, prefix)





def features_step1(sub_graph_prefix,
                   data_path, data_key,
                   labels_path, labels_key,
                   offset_file, block_file,
                   out_path):

    t0 = time.time()
    block_list = np.load(block_file).tolist()
    with open(offset_file, 'r') as f:
        offsets = json.load(f)

    dtype = z5py.File(data_path)[data_key].dtype

    if offsets is not None:
        affinity_function = ndist.extractBlockFeaturesFromAffinityMaps_uint8 if dtype == 'uint8' else \
            ndist.extractBlockFeaturesFromAffinityMaps_float32

        affinity_function(sub_graph_prefix,
                          data_path, data_key,
                          labels_path, labels_key,
                          block_list, os.path.join(out_path, 'blocks'),
                          offsets)
    else:
        boundary_function = ndist.extractBlockFeaturesFromBoundaryMaps_uint8 if dtype == 'uint8' else \
            ndist.extractBlockFeaturesFromBoundaryMaps_float32
        boundary_function(sub_graph_prefix,
                          data_path, data_key,
                          labels_path, labels_key,
                          block_list,
                          os.path.join(out_path, 'blocks'))

    job_id = int(os.path.split(block_file)[1].split('_')[2][:-4])
    print("Success job %i" % job_id)
    print("In %f s" % (time.time() - t0,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sub_graph_prefix", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("data_key", type=str)
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)
    parser.add_argument("--offset_file", type=str)
    parser.add_argument("--block_file", type=str)
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    features_step1(args.sub_graph_prefix,
                   args.data_path, args.data_key,
                   args.labels_path, args.labels_key,
                   args.offset_file, args.block_file,
                   args.out_path)
