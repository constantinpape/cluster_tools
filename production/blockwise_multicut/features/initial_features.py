#! /usr/bin/python

import os
import argparse
import time
import json
import subprocess
from concurrent import futures

import nifty
import nifty.distributed as ndist
import z5py
import luigi


class InitialFeaturesTask(luigi.Task):
    """
    Compute initial sub-graphs
    """

    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    ws_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    out_path = luigi.Parameter()
    max_jobs = luigi.Parameter()
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def _prepare_jobs(self, n_jobs, n_blocks, offsets):
        block_list = list(range(n_blocks))
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'affinity_offsets': offsets,
                          'block_list': block_jobs}
            config_path = os.path.join(self.tmp_folder, 'initial_features_config_job%i.json' % job_id)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id):
        script_path = os.path.join(self.tmp_folder, 'initial_features.py')
        config_path = os.path.join(self.tmp_folder, 'initial_features_config_job%i.json' % job_id)
        subgraph_prefix = os.path.join(self.graph_path, "sub_graphs/s0/block_")
        command = '%s %s %s %s %s %s %i %s %s' % (script_path, self.path, self.aff_key, self.ws_key,
                                                  subgraph_prefix,
                                                  self.out_path, job_id, config_path,
                                                  self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_initial_features_%i' % job_id)
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_initial_features_%i.err' % job_id)
        bsub_command = 'bsub -J initial_features_%i -We %i -o %s -e %s \'%s\'' % (job_id,
                                                                                  self.time_estimate,
                                                                                  log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _collect_outputs(self, n_jobs):
        times = []
        processed_jobs = []
        for job_id in range(n_jobs):
            res_file = os.path.join(self.tmp_folder, 'initial_features_job%i.json' % job_id)
            try:
                with open(res_file) as f:
                    res = json.load(f)
                times.append(res['t'])
                processed_jobs.append(job_id)
                os.remove(res_file)
            except Exception:
                continue
        return processed_jobs, times

    def run(self):
        from production import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'initial_features.py'),
                              os.path.join(self.tmp_folder, 'initial_features.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            offsets = config['affinity_offsets']
            # TODO support computation with roi
            if 'roi' in config:
                roi = config['roi']
            else:
                roi = None

        # hardconded keys
        graph_key = 'graph'
        out_key = 'features'

        # create the outpuy files
        f_graph = z5py.File(self.graph_path, use_zarr_format=False)
        shape = f_graph.attrs['shape']
        ds_graph = f_graph[graph_key]
        n_edges = ds_graph.attrs['numberOfEdges']

        f_out = z5py.File(self.out_path, use_zarr_format=False)
        f_out.require_group('blocks')
        # chunk size = 64**3
        chunk_size = min(262144, n_edges)
        f_out.require_dataset(out_key, dtype='float64', shape=(n_edges, 10),
                              chunks=(chunk_size, 1), compression='gzip')

        # get number of blocks
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks
        # find the actual number of jobs and prepare job configs
        n_jobs = min(n_blocks, self.max_jobs)
        self._prepare_jobs(n_jobs, n_blocks, offsets)

        # submit the jobs
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(n_jobs) as tp:
                tasks = [tp.submit(self._submit_job, job_id) for job_id in range(n_jobs)]
                [t.result() for t in tasks]
        else:
            for job_id in range(n_jobs):
                self._submit_job(job_id)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        # check the job outputs
        processed_jobs, times = self._collect_outputs(n_jobs)
        assert len(processed_jobs) == len(times)
        success = len(processed_jobs) == n_jobs

        # write output file if we succeed, otherwise write partial
        # success to different file and raise exception
        if success:
            out = self.output()
            # TODO does 'out' support with job?
            fres = out.open('w')
            json.dump({'times': times}, fres)
            fres.close()
        else:
            log_path = os.path.join(self.tmp_folder, 'initial_features_partial.json')
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_jobs': processed_jobs}, out)
            raise RuntimeError("InitialFeatureTask failed, %i / %i jobs processed, serialized partial results to %s" % (len(processed_jobs),
                                                                                                                        n_jobs,
                                                                                                                        log_path))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'initial_features.log'))


def initial_features(path, aff_key, labels_key,
                     sub_graph_prefix, out_path,
                     job_id, config_path, tmp_folder):

    t0 = time.time()
    with open(config_path, 'r') as f:
        config = json.load(f)
        offsets = config['affinity_offsets']
        block_list = config['block_list']

    dtype = z5py.File(path)[aff_key].dtype

    if offsets is not None:
        affinity_function = ndist.extractBlockFeaturesFromAffinityMaps_uint8 if dtype == 'uint8' else \
            ndist.extractBlockFeaturesFromAffinityMaps_float32

        affinity_function(sub_graph_prefix,
                          path, aff_key,
                          path, labels_key,
                          block_list, os.path.join(out_path, 'blocks'),
                          offsets)
    else:
        boundary_function = ndist.extractBlockFeaturesFromBoundaryMaps_uint8 if dtype == 'uint8' else \
            ndist.extractBlockFeaturesFromBoundaryMaps_float32
        boundary_function(sub_graph_prefix,
                          path, aff_key,
                          path, labels_key,
                          block_list,
                          os.path.join(out_path, 'blocks'))

    res_file = os.path.join(tmp_folder, 'initial_features_job%i.json' % job_id)
    with open(res_file, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("aff_key", type=str)
    parser.add_argument("labels_key", type=str)
    parser.add_argument("sub_graph_prefix", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("job_id", type=int)
    parser.add_argument("config_path", type=str)
    parser.add_argument("tmp_folder", type=str)
    args = parser.parse_args()

    initial_features(args.path, args.aff_key,
                     args.labels_key, args.sub_graph_prefix,
                     args.out_path, args.job_id,
                     args.config_path, args.tmp_folder)
