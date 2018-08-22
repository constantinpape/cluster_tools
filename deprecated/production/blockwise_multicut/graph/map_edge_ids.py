#! /usr/bin/python

import os
import argparse
import time
import subprocess
import json
from concurrent import futures

import z5py
import nifty
import nifty.distributed as ndist
import luigi


class MapEdgesTask(luigi.Task):
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

    def _prepare_job(self, scale, config):
        config_path = os.path.join(self.tmp_folder, 'map_edge_ids_s%i.json' % scale)
        with open(config_path, 'w') as f:
            json.dump(config, f)

    def _submit_job(self, scale, n_threads):
        # make commands and submit the job
        script_path = os.path.join(self.tmp_folder, 'map_edge_ids.py')
        config_path = os.path.join(self.tmp_folder, 'map_edge_ids_s%i.json' % scale)
        command = '%s %s %i %s %s' % (script_path, self.out_path, scale,
                                      config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_map_edge_ids_s%i' % scale)
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_map_edge_ids_s%i' % scale)
        bsub_command = 'bsub -n %i -J map_edge_ids_s%i -We %i -o %s -e %s \'%s\'' % (n_threads, scale,
                                                                                     self.time_estimate,
                                                                                     log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _collect_outputs(self, n_scales):
        times = []
        processed_scales = []
        for scale in range(n_scales):
            res_file = os.path.join(self.tmp_folder, 'mape_edge_ids_s%i' % scale)
            try:
                with open(res_file) as f:
                    res = json.load(f)
                times.append(res['t'])
                processed_scales.append(scale)
                os.remove(res_file)
            except Exception:
                continue
        return processed_scales, times

    def run(self):
        from production import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(self.tmp_folder, 'map_edge_ids.py')
        util.copy_and_replace(os.path.join(file_dir, 'map_edge_ids.py'),
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
        for scale in range(self.max_scale + 1):
            self._prepare_job(scale, config)

        # submit the jobs
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(self.max_scale + 1) as tp:
                tasks = [tp.submit(self._submit_job, scale, n_threads)
                         for scale in range(self.max_scale + 1)]
                [t.result() for t in tasks]
        else:
            for scale in range(self.max_scale + 1):
                self._submit_job(scale, n_threads)

        if not self.run_local:
            util.wait_for_jobs('papec')

        # check for results
        processed_scales, times = self._collect_outputs(self.max_scale + 1)
        success = len(processed_scales) == self.max_scale + 1
        if success:
            with open(self.output().path, 'w') as f:
                json.dump({'times': times}, f)
        else:
            log_path = os.path.join(self.tmp_folder, 'map_edge_ids_partial.log')
            with open(log_path, 'w') as f:
                json.dump({'processed_scales': processed_scales, 'times': times}, f)
            raise RuntimeError("MapEdgesTask failed for %i / %i scales," % (len(times), self.max_scale + 1) +
                               "partial results serialized to %s" % log_path)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'map_edge_ids.log'))


def map_edge_ids(graph_path, scale, config_path, tmp_folder):

    t0 = time.time()
    with open(config_path) as f:
        config = json.load(f)
        initial_block_shape = config['block_shape']
        n_threads = config['n_threads']
        roi = config.get('roi', None)
    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    f_graph = z5py.File(graph_path)
    shape = f_graph.attrs['shape']
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    input_key = 'graph'

    block_prefix = 'sub_graphs/s%i/block_' % scale
    if roi is None:
        n_blocks = blocking.numberOfBlocks
        ndist.mapEdgeIdsForAllBlocks(graph_path, input_key,
                                     blockPrefix=block_prefix,
                                     numberOfBlocks=n_blocks,
                                     numberOfThreads=n_threads)

    else:
        block_list = blocking.getBlockIdsOverlappingBoundingBox(roi[0],
                                                                roi[1],
                                                                [0, 0, 0]).tolist()
        ndist.mapEdgeIds(graph_path, input_key,
                         blockPrefix=block_prefix,
                         blockIds=block_list,
                         numberOfThreads=n_threads)

    res_file = os.path.join(tmp_folder, 'mape_edge_ids_s%i' % scale)
    with open(res_file, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("config_path", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()

    map_edge_ids(args.graph_path,
                 args.scale,
                 args.config_path,
                 args.tmp_folder)
