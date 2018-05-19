#! /usr/bin/python

import os
import argparse
import json
import numpy as np
import subprocess
import luigi


class OffsetTask(luigi.Task):
    """
    Run the offset computation
    """

    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def run(self):
        from .. import util

        # get input and output path
        in_path = self.input().path
        out_path = self.output().path

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'compute_block_offsets.py'),
                              os.path.join(self.tmp_folder, 'compute_block_offsets.py'))

        # assemble the commands
        script_path = os.path.join(self.tmp_folder, 'compute_block_offsets.py')
        command = '%s %s %s' % (script_path, in_path, out_path)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_compute_offsets.log')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_compute_offsets.err')
        bsub_command = 'bsub -J compute_offsets -We %i -o %s -e %s \'%s\'' % (self.time_estimate,
                                                                              log_file, err_file, command)

        # submit job
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

        # wait job is finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        # check for correct execution
        success = os.path.exists(out_path)
        if not success:
            raise RuntimeError("Compute offsets failed")

    def output(self):
        out_file = os.path.join(self.tmp_folder, 'block_offsets.json')
        return luigi.LocalTarget(out_file)


def compute_block_offsets(input_file, output_file):
    with open(input_file) as f:
        offsets = np.array(json.load(f)['n_components'], dtype='uint64')

    empty_blocks = np.where(offsets == 0)[0]

    last_offset = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    n_labels = int(offsets[-1] + last_offset + 1)

    with open(output_file, 'w') as f:
        json.dump({'offsets': offsets.tolist(),
                   'empty_blocks': empty_blocks.tolist(),
                   'n_labels': n_labels}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()
    compute_block_offsets(args.input_file, args.output_file)
