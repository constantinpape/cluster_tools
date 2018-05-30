#! /usr/bin/python

import os
import argparse
import time
import subprocess
import json

import z5py
import nifty
import nifty.distributed as ndist
import luigi


class MergeGraphTask(luigi.Task):
    """
    Merge complete graph
    """

    out_path = luigi.Parameter()
    max_scale = luigi.IntParameter()
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
        file_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(self.tmp_folder, 'merge_graph.py')
        util.copy_and_replace(os.path.join(file_dir, 'merge_graph.py'),
                              script_path)

        with open(self.config_path) as f:
            config = json.load(f)
            init_block_shape = config['block_shape']
            n_threads = config['n_threads']
            roi = config.get('roi', None)

        # make config for the job
        config = {'block_shape': init_block_shape,
                  'n_threads': n_threads,
                  'roi': roi}
        config_path = os.path.join(self.tmp_folder, 'merge_graph.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # make commands and submit the job
        command = '%s %s %i %s %s' % (script_path, self.out_path, self.max_scale,
                                      config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_merge_graph')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_merge_graph')
        bsub_command = 'bsub -n %i -J merge_graph -We %i -o %s -e %s \'%s\'' % (n_threads, self.time_estimate,
                                                                                log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)
            util.wait_for_jobs('papec')

        res_file = self.output().path
        try:
            with open(res_file) as f:
                json.load(f)['t']
            success = True
        except Exception:
            success = False

        if not success:
            raise RuntimeError("MergeGraphTask failed")

    def output(self):
        res_file = os.path.join(self.tmp_folder, 'log_merge_graph.log')
        return luigi.LocalTarget(res_file)


def merge_graph(graph_path, last_scale, config_file, tmp_folder):

    t0 = time.time()
    with open(config_file) as f:
        config = json.load(f)
        initial_block_shape = config['block_shape']
        n_threads = config['n_threads']
        roi = config.get('roi', None)

    factor = 2**last_scale
    block_shape = [factor * bs for bs in initial_block_shape]

    f_graph = z5py.File(graph_path)
    shape = f_graph.attrs['shape']
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)

    if roi is None:
        block_list = list(range(blocking.numberOfBlocks))
    else:
        block_list = blocking.getBlockIdsOverlappingBoundingBox(roi[0],
                                                                roi[1],
                                                                [0, 0, 0]).tolist()

    block_prefix = 'sub_graphs/s%i/block_' % last_scale
    output_key = 'graph'
    ndist.mergeSubgraphs(graph_path,
                         blockPrefix=block_prefix,
                         blockIds=block_list,
                         outKey=output_key,
                         numberOfThreads=n_threads)
    f_graph[output_key].attrs['shape'] = shape
    res_file = os.path.join(tmp_folder, 'log_merge_graph.log')
    with open(res_file, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("last_scale", type=int)
    parser.add_argument("config_file", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()

    merge_graph(args.graph_path,
                args.last_scale,
                args.config_file,
                args.tmp_folder)
