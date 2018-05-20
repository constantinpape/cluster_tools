#! /usr/bin/python

import os
import json
import time
import argparse
import pickle
import subprocess

import numpy as np
import vigra
import z5py
import nifty
import luigi


class FindLabelingTask(luigi.Task):
    """
    """

    path = luigi.Parameter()
    key = luigi.Parameter()
    # path to the configuration
    # TODO allow individual paths for individual blocks
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def _submit_job(self, block_shape):
        script_path = os.path.join(self.tmp_folder, 'find_labeling.py')
        assert os.path.exists(script_path)
        command = '%s %s %s %s %s' % (script_path, self.path, self.key,
                                      self.tmp_folder, ' '.join(map(str, block_shape)))
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_find_labeling')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_find_abeling')
        bsub_command = 'bsub -J find_labeling -We %i -o %s -e %s \'%s\'' % (self.time_estimate,
                                                                            log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'find_labeling.py'),
                              os.path.join(self.tmp_folder, 'find_labeling.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            # TODO support computation with roi
            if 'roi' in config:
                have_roi = True

        self._submit_job(block_shape)
        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        try:
            out_path = self.output().path
            with open(out_path) as f:
                json.load(f)['t']
            success = True
        except Exception:
            success = False

        if not success:
            raise RuntimeError("FindLabelingTask failed")

    def output(self):
        out_file = os.path.join(self.tmp_folder, 'relabeling.pkl')
        return luigi.LocalTarget(out_file)


# TODO this could be parallelized
def find_labeling(labels_path, labels_key, tmp_folder, block_shape, n_threads=1):
    t0 = time.time()
    ds_labels = z5py.File(labels_path, use_zarr_format=False)[labels_key]
    shape = ds_labels.shape

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)

    uniques = np.concatenate([np.load(os.path.join(tmp_folder, 'uniques_block_%i.npy' % block_id))
                              for block_id in range(blocking.numberOfBlocks)])
    uniques = np.unique(uniques).astype('uint64', copy=False)
    _, max_id, mapping = vigra.analysis.relabelConsecutive(uniques, keep_zeros=True, start_label=1)

    ds_labels.attrs['maxId'] = max_id

    with open(os.path.join(tmp_folder, 'relabeling.pkl'), 'wb') as f:
        pickle.dump(mapping, f)
    res_path = os.path.join(tmp_folder, 'relabeling_time.json')
    with open(res_path, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)

    parser.add_argument("tmp_folder", type=str)
    parser.add_argument("block_shape", nargs=3, type=int)
    parser.add_argument("--n_threads", type=int, default=1)

    args = parser.parse_args()
    find_labeling(args.labels_path, args.labels_key,
                  args.tmp_folder,
                  list(args.block_shape),
                  args.n_threads)
