#! /usr/bin/python

import time
import os
import json
import argparse
import subprocess
from concurrent import futures

import numpy as np
import nifty
import z5py
import luigi


class FindUniquesTask(luigi.Task):
    """
    """

    path = luigi.Parameter()
    key = luigi.Parameter()
    max_jobs = luigi.IntParameter()
    # path to the configuration
    # TODO allow individual paths for individual blocks
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependecy = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def _prepare_jobs(self, n_jobs, n_blocks, block_shape):
        block_list = list(range(n_blocks))
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_list': block_jobs,
                          'block_shape': block_shape}
            config_path = os.path.join(self.tmp_folder, 'find_uniques_config_job%i.json' % job_id)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id):
        script_path = os.path.join(self.tmp_folder, 'find_uniques.py')
        assert os.path.exists(script_path)
        config_path = os.path.join(self.tmp_folder, 'find_uniques_config_job%i.json' % job_id)
        command = '%s %s %s %s %s' % (script_path, self.path, self.key,
                                      config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_uniques_block_%i' % job_id)
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_uniques_block_%i.err' % job_id)
        bsub_command = 'bsub -J uniques_block_%i -We %i -o %s -e %s \'%s\'' % (job_id,
                                                                               self.time_estimate,
                                                                               log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _collect_outputs(self, n_blocks):
        times = []
        processed_blocks = []
        for block_id in range(n_blocks):
            res_file = os.path.join(self.tmp_folder, 'times_uniques_block%i.json' % block_id)
            try:
                with open(res_file) as f:
                    res = json.load(f)
                    times.append(res['t'])
                processed_blocks.append(block_id)
                os.remove(res_file)
            except Exception:
                continue
        return processed_blocks, times

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'find_uniques.py'),
                              os.path.join(self.tmp_folder, 'find_uniques.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            # TODO support computation with roi
            if 'roi' in config:
                have_roi = True

        # find the shape and number of blocks
        shape = z5py.File(self.path)[self.key].shape
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks

        # find the actual number of jobs and prepare job configs
        n_jobs = min(n_blocks, self.max_jobs)
        self._prepare_jobs(n_jobs, n_blocks, block_shape)

        # submit the jobs
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(n_jobs) as tp:
                tasks = [tp.submit(self._submit_job, job_id)
                         for job_id in range(n_jobs)]
                [t.result() for t in tasks]
        else:
            for job_id in range(n_jobs):
                self._submit_job(job_id)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        # check the job outputs
        processed_blocks, times = self._collect_outputs(n_blocks)
        assert len(processed_blocks) == len(times)
        success = len(processed_blocks) == n_blocks

        # write output file if we succeed, otherwise write partial
        # success to different file and raise exception
        if success:
            out = self.output()
            # TODO does 'out' support with block?
            fres = out.open('w')
            json.dump({'times': times}, fres)
            fres.close()
        else:
            log_path = os.path.join(self.tmp_folder, 'find_uniques_partial.json')
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("FindUniquesTask failed, %i / %i blocks processed, serialized partial results to %s" % (len(processed_blocks),
                                                                                                                       n_blocks,
                                                                                                                       log_path))

    def output(self):
        out_file = os.path.join(self.tmp_folder, 'find_uniques.json')
        return luigi.LocalTarget(out_file)


def uniques_in_block(block_id, blocking, ds, tmp_folder):
    t0 = time.time()
    block = blocking.getBlock(block_id)
    bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
    labels = ds[bb]
    uniques = np.unique(labels)
    np.save(os.path.join(tmp_folder, 'uniques_block_%i.npy' % block_id), uniques)
    res_path = os.path.join(tmp_folder, 'times_uniques_block%i.json' % block_id)
    with open(res_path, 'w') as f:
        f.dump({'t': time.time() - t0})


def find_uniques(labels_path, labels_key, tmp_folder, config_file):
    ds_labels = z5py.File(labels_path, use_zarr_format=False)[labels_key]
    shape = ds_labels.shape

    with open(config_file) as f:
        config = json.load(f)
        block_list = config['block_list']
        block_shape = config['block_shape']

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)

    [uniques_in_block(block_id, blocking, ds_labels, tmp_folder) for block_id in block_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)

    parser.add_argument("tmp_folder", type=str)
    parser.add_argument("config_file", type=str)

    args = parser.parse_args()
    find_uniques(args.labels_path, args.labels_key,
                 args.tmp_folder,
                 args.config_file)
