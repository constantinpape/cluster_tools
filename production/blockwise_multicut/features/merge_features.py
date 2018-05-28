#! /usr/bin/python

import os
import argparse
import time
import json
import subprocess
# from concurrent import futures

import nifty
import nifty.distributed as ndist
import z5py
import luigi


class MergeFeaturesTask(luigi.Task):
    """
    Merge features
    """

    graph_path = luigi.Parameter()
    out_path = luigi.Parameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def run(self):
        from production import util

        # copy the script to the temp folder and replace the shebang
        script_path = os.path.join(self.tmp_folder, 'merge_features.py')
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'merge_features.py'),
                              script_path)

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            n_threads = config['n_threads']
            # TODO support computation with roi
            if 'roi' in config:
                roi = config['roi']
            else:
                roi = None

        # write job config
        job_config = {'block_shape': block_shape, 'n_threads': n_threads}
        config_path = os.path.join(self.tmp_folder, 'merge_features_config.json')
        with open(config_path, 'w') as f:
            json.dump(job_config, f)

        # submit job
        command = '%s %s %s %s %s' % (script_path, self.graph_path, self.out_path,
                                      config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_merge_features')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_merge_features')
        bsub_command = 'bsub -n %i -J merge_features -We %i -o %s -e %s \'%s\'' % (n_threads, self.time_estimate,
                                                                                   log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)
            util.wait_for_jobs('papec')

        try:
            with open(self.output().path) as f:
                json.load(f)['t']
            success = True
        except Exception:
            success = False

        if not success:
            raise RuntimeError("MergeFeaturesTask failed")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'merge_features.log'))


def merge_features(graph_path, out_path, config_path, tmp_folder):
    t0 = time.time()
    edge_begin = 0
    out_key = 'features'
    edge_end = z5py.File(out_path)[out_key].shape[0]

    with open(config_path) as f:
        config = json.load(f)
        block_shape = config['block_shape']
        n_threads = config['n_threads']
    shape = z5py.File(graph_path).attrs['shape']
    blocking = nifty.tools.blocking([0, 0, 0],
                                    shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    subgraph_prefix = os.path.join(graph_path, 'sub_graphs', 's0', 'block_')
    print(subgraph_prefix)
    features_tmp_prefix = os.path.join(out_path, 'blocks', 'block_')
    ndist.mergeFeatureBlocks(subgraph_prefix,
                             features_tmp_prefix,
                             os.path.join(out_path, out_key),
                             numberOfBlocks=n_blocks,
                             edgeIdBegin=edge_begin,
                             edgeIdEnd=edge_end,
                             numberOfThreads=n_threads)

    res_file = os.path.join(tmp_folder, 'merge_features.log')
    with open(res_file, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("config_path", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()

    merge_features(args.graph_path,
                   args.out_path,
                   args.config_path,
                   args.tmp_folder)
