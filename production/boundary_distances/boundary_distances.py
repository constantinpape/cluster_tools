#! /usr/bin/python

import os
import time
import argparse
import subprocess
import json
from concurrent import futures

import numpy as np
import vigra

import luigi
import nifty
import z5py

from production.util import DummyTask


class BoundaryDistanceTask(luigi.Task):
    path = luigi.Parameter()
    seg_key = luigi.Parameter()
    out_key = luigi.Parameter()
    id_list = luigi.ListParameter()
    max_workers = luigi.IntParameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter(default=DummyTask())
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    # TODO allow different configs for different blocks
    def _prepare_jobs(self, n_jobs, n_blocks, config):
        block_list = list(range(n_blocks))
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_list': block_jobs, **config}
            config_path = os.path.join(self.tmp_folder, 'boundary_disance_config_job%i.json' % job_id)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id):
        script_path = os.path.join(self.tmp_folder, 'boundary_distances.py')
        assert os.path.exists(script_path)
        config_path = os.path.join(self.tmp_folder, 'boundary_distances_config_job%i.json' % job_id)
        command = '%s %s %s %s %s %s' % (script_path, self.path, self.seg_key, self.out_key,
                                         config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_boundary_distances_job_%i' % job_id)
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_boundary_distances_job_%i.err' % job_id)
        bsub_command = 'bsub -J boundary_distances_job_%i -We %i -o %s -e %s \'%s\'' % (job_id,
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
            log_path = os.path.join(self.tmp_folder,
                                    'log_boundary_distances_block%i.json' % block_id)
            try:
                with open(log_path) as f:
                    res = json.load(f)
                    times.append(res['t'])
                processed_blocks.append(block_id)
                os.remove(log_path)
            except Exception:
                continue
        return processed_blocks, times

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        util.copy_and_replace(os.path.join(file_dir, 'boundary_distances.py'),
                              os.path.join(self.tmp_folder, 'boundary_distances.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            chunks = tuple(config['chunks'])
            # TODO support computation with roi
            # if 'roi' in config:
            #     have_roi = True

            # load task specific parameters
            resolution = config['resolution']
            # the halo for the distance transform computation
            # use a 400 nm halo by default
            halo = config.get('halo', [10, 100, 100])

        # find the shape and number of blocks
        f = z5py.File(self.path)
        shape = f[self.seg_key].shape
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks

        # make the output dataset
        f.require_dataset(self.out_key, shape=shape,
                          chunks=chunks, dtype='float32', compression='gzip')

        # find the actual number of jobs and prepare job configs
        n_jobs = min(n_blocks, self.max_jobs)
        out_config = {'resolution': resolution,
                      'block_shape': block_shape,
                      'id_list': self.id_list,
                      'halo': halo}
        self._prepare_jobs(n_jobs, n_blocks, out_config)

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
            log_path = os.path.join(self.tmp_folder, 'boundary_distances_partial.json')
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("BoundaryDistancesask failed, %i / %i blocks processed," % (len(processed_blocks),
                                                                                           n_blocks) +
                               "serialized partial results to %s" % log_path)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'boundary_distances.log'))


def boundary_distances_block(block_id, blocking,
                             ds_seg, ds_out,
                             resolution, id_list,
                             halo, tmp_folder):

    t0 = time.time()
    log_path = os.path.join(tmp_folder, 'log_boundary_distances_block%i.json' % block_id)

    block = blocking.getBlockWithHalo(block_id, halo)
    outer_bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
    inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
    local_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))

    seg = ds_seg[outer_bb]
    inner_seg = seg[inner_bb]

    # check if any of the ids of interest are in the segmentation
    sub_ids = np.unique(inner_seg)
    sub_ids = sub_ids[np.in1d(sub_ids, id_list)]
    # if we don't find any of the ids, continue
    if len(sub_ids) == 0:
        with open(log_path, 'w') as f:
            json.dump({'t': time.time() - t0}, f)

    # FIXME would be nice to have a 3d fucntion in vigra
    # alternatively, we could also shift the segmentation
    # along the 3 axesand substract
    # (currently we don't treat z boundaries correctly)
    # compute the region boundaries
    boundaries = np.zeros_like(seg, dtype='uint8')
    for z in range(seg.shape[0]):
        boundaries[z] = vigra.analysis.regionImageToEdgeImage(seg[z]).astype('uint8')

    # compute the distance trafo
    dt = vigra.filters.distanceTransform(boundaries, pixel_pitch=resolution)

    # calculate maxima of the distance transform
    maxima = vigra.analysis.localMaxima3D(dt, marker=1,
                                          allowAtBorder=True, allowPlateaus=True).astype('uint8')

    # iterate over the objects of interest in the subvolume and write out their boundary distance
    output = np.zeros_like(inner_seg, dtype='float32')
    for sub_id in sub_ids:
        id_mask = seg == sub_id
        non_id_mask = np.logical_not(id_mask)

        this_dt = maxima.copy()
        this_dt[non_id_mask] = 0
        vigra.filters.distanceTransform(this_dt, pixel_pitch=resolution, output=this_dt)

        id_mask = id_mask[local_bb]
        output[id_mask] = this_dt[local_bb]

    ds_out[inner_bb] = output
    with open(log_path, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


def boundary_distances(path, seg_key, out_key, config_path, tmp_folder):
    # load the configuration
    with open(config_path) as f:
        config = json.load(f)
        block_ids = config['block_list']
        block_shape = config['block_shape']
        resolution = config['resolution']
        id_list = config['id_list']
        # the halo for the distance transform computation
        # use a 400 nm halo by default
        halo = config.get('halo', [10, 100, 100])

    f = z5py.File(path)
    ds_seg = f[seg_key]
    ds_out = f[out_key]
    shape = ds_seg.shape
    blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)

    for block_id in block_ids:
        boundary_distances_block(block_id, blocking,
                                 ds_seg, ds_out,
                                 resolution, id_list,
                                 halo, tmp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('seg_key', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('config_path', type=str)
    parser.add_argument('tmp_folder', type=str)

    args = parser.parse_args()
    boundary_distances(args.path, args.seg_key,
                       args.out_keys, args.config_path,
                       args.tmp_folder)
