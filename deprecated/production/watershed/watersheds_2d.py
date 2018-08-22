#! /usr/bin/python

import time
import os
import argparse
import json
import subprocess
from concurrent import futures

import numpy as np
import z5py
import vigra
import nifty
import luigi


# TODO more clean up (job config files)
# TODO computation with rois
class Watershed2dTask(luigi.Task):
    """
    Run watersheds to fill up components
    """

    # path to the n5 file and keys
    path = luigi.Parameter()
    aff_key = luigi.Parameter()
    mask_key = luigi.Parameter()
    out_key = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    # path to the configuration
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)
    # TODO optional parameter to just run a subset of blocks

    # TODO there must be a more efficient way to do this
    def _make_checkerboard(self, blocking):
        blocks_a = [0]
        blocks_b = []
        all_blocks = [0]

        def recurse(current_block, insert_list):
            other_list = blocks_a if insert_list is blocks_b else blocks_b
            for dim in range(3):
                ngb_id = blocking.getNeighborId(current_block, dim, False)
                if ngb_id != -1:
                    if ngb_id not in all_blocks:
                        insert_list.append(ngb_id)
                        all_blocks.append(ngb_id)
                        recurse(ngb_id, other_list)

        recurse(0, blocks_b)
        all_blocks = blocks_a + blocks_b
        expected = set(range(blocking.numberOfBlocks))
        assert len(all_blocks) == len(expected), "%i, %i" % (len(all_blocks), len(expected))
        assert len(set(all_blocks) - expected) == 0
        assert len(blocks_a) == len(blocks_b), "%i, %i" % (len(blocks_a), len(blocks_b))
        return blocks_a, blocks_b

    def _prepare_jobs(self, n_jobs, block_list, config, prefix):
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'config': config,
                          'block_list': block_jobs}
            config_path = os.path.join(self.tmp_folder, 'watershed2d_config_%s_job%i.json' % (prefix,
                                                                                              job_id))
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _submit_job(self, job_id, prefix):
        script_path = os.path.join(self.tmp_folder, 'watersheds_2d.py')
        assert os.path.exists(script_path)
        config_path = os.path.join(self.tmp_folder, 'watershed2d_config_%s_job%i.json' % (prefix,
                                                                                          job_id))
        command = '%s %s %s %s %s %i %s %s' % (script_path, self.path, self.aff_key,
                                               self.mask_key, self.out_key,
                                               job_id, config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_watershed2d_%s_%i' % (prefix, job_id))
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_watershed2d_%s_%i.err' % (prefix, job_id))
        bsub_command = 'bsub -J filling_watershed_%i -We %i -o %s -e %s \'%s\'' % (job_id,
                                                                                   self.time_estimate,
                                                                                   log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

    def _submit_jobs(self, n_jobs, prefix):
        from .. import util
        if self.run_local:
            # this only works in python 3 ?!
            with futures.ProcessPoolExecutor(n_jobs) as tp:
                tasks = [tp.submit(self._submit_job, job_id, prefix)
                         for job_id in range(n_jobs)]
                [t.result() for t in tasks]
        else:
            for job_id in range(n_jobs):
                self._submit_job(job_id, prefix)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

    def _collect_outputs(self, n_blocks):
        times = []
        processed_blocks = []
        for block_id in range(n_blocks):
            res_file = os.path.join(self.tmp_folder, 'watershed_2d_block%i.json' % block_id)
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
        util.copy_and_replace(os.path.join(file_dir, 'watersheds_2d.py'),
                              os.path.join(self.tmp_folder, 'watersheds_2d.py'))

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            chunks = tuple(config['chunks'])
            # TODO support computation with roi
            if 'roi' in config:
                have_roi = True

        # get the shape
        f5 = z5py.File(self.path)
        shape = f5[self.mask_key].shape

        # require the output dataset
        f5.require_dataset(self.out_key, shape=shape, chunks=chunks,
                           compression='gzip', dtype='uint64')

        # get the blocking
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        # divide the blocks in 2 parts, defining a checkerboard pattern
        blocks_a, blocks_b = self._make_checkerboard(blocking)

        # find the actual number of jobs and prepare job configs
        n_jobs = min(len(blocks_a), self.max_jobs)
        self._prepare_jobs(n_jobs, blocks_a, config, 'a')
        # add halo to config for second block list
        config.update({'second_pass_ws2d': True})
        self._prepare_jobs(n_jobs, blocks_b, config, 'b')

        # submit the jobs
        print("Start blocks a")
        self._submit_jobs(n_jobs, 'a')
        print("Start blocks b")
        self._submit_jobs(n_jobs, 'b')

        n_blocks = blocking.numberOfBlocks
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
            log_path = os.path.join(self.tmp_folder, 'watersheds_2d_partial.json')
            with open(log_path, 'w') as out:
                json.dump({'times': times,
                           'processed_blocks': processed_blocks}, out)
            raise RuntimeError("Watershed2dTask failed, %i / %i blocks processed, serialized partial results to %s" % (len(processed_blocks),
                                                                                                                       n_blocks,
                                                                                                                       log_path))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'watershed_2d.log'))


def compute_max_seeds(hmap, boundary_threshold,
                      sigma, offset, mask,
                      initial_seeds=None):

    # we only compute the seeds on the smaller crop of the volume
    seeds = np.zeros_like(hmap, dtype='uint64')

    for z in range(seeds.shape[0]):
        # compute distance transform on the full 2d plane
        dtz = vigra.filters.distanceTransform((hmap[z] > boundary_threshold).astype('uint32'))
        if sigma > 0:
            vigra.filters.gaussianSmoothing(dtz, sigma, out=dtz)

        # compute local maxima of the distance transform, then crop
        seeds_z = vigra.analysis.localMaxima(dtz, allowPlateaus=True, allowAtBorder=True, marker=np.nan)
        seeds_z = vigra.analysis.labelImageWithBackground(np.isnan(seeds_z).view('uint8'))

        seeds_z[mask[z]] = int(seeds_z.max()) + 1
        # add offset to the seeds
        seeds_z = seeds_z.astype('uint64')
        seeds_z[seeds_z != 0] += offset
        offset = int(seeds_z.max()) + 1

        # check if we have initial seeds
        # and add them up if we do (we should be guranteed that the ids do not overlap by
        # the block ovelap added)
        if initial_seeds is not None:
            initial_seed_mask = initial_seeds[z] != 0
            # print(seeds_z.shape, initial_seed_mask.shape, initial_seeds[z].shape)
            seeds_z[initial_seed_mask] = initial_seeds[z][initial_seed_mask]

        # write seeds to the corresponding slice
        seeds[z] = seeds_z
    return seeds


def run_2d_ws(hmap, seeds, mask, size_filter):
    # iterate over the slices
    for z in range(seeds.shape[0]):

        # we need to remap the seeds consecutively, because vigra
        # watersheds can only handle uint32 seeds, and we WILL overflow uint32
        seeds_z, _, old_to_new = vigra.analysis.relabelConsecutive(seeds[z],
                                                                   start_label=1,
                                                                   keep_zeros=True)
        new_to_old = {new: old for old, new in old_to_new.items()}
        ws_z = vigra.analysis.watershedsNew(hmap[z], seeds=seeds_z.astype('uint32'))[0]

        # apply size_filter
        if size_filter > 0:
            ids, sizes = np.unique(ws_z, return_counts=True)
            filter_ids = ids[sizes < size_filter]
            # do not filter ids that belong to the extended seeds
            filter_mask = np.ma.masked_array(ws_z, np.in1d(ws_z, filter_ids)).mask
            ws_z[filter_mask] = 0
            vigra.analysis.watershedsNew(hmap[z], seeds=ws_z, out=ws_z)

        # set the invalid mask to zero
        ws_z[mask[z]] = 0

        # map bad to original ids
        ws_z = ws_z.astype('uint64')
        ws_z = nifty.tools.takeDict(new_to_old, ws_z)

        # write the watershed to the seeds
        seeds[z] = ws_z
    return seeds, int(seeds.max())


def ws_block(ds_affs, ds_mask, ds_out,
             blocking, block_id, block_config,
             tmp_folder):

    print("Processing block", block_id)
    res_file = os.path.join(tmp_folder, 'watershed_2d_block%i.json' % block_id)

    t0 = time.time()

    # get offset to make new seeds unique between blocks
    # (we need to relabel later to make processing efficient !)
    offset = block_id * np.prod(blocking.blockShape)

    boundary_threshold = block_config['boundary_threshold']
    sigma_maxima = block_config['sigma_maxima']
    size_filter = block_config['size_filter']
    # check if we are in the second pass (which means we already have seeds)
    second_pass = 'second_pass_ws2d' in block_config

    # halo is hard-coded for now / 75 pixel should be enough
    halo = [0, 100, 100]
    block = blocking.getBlockWithHalo(block_id, halo)

    outer_bb = tuple(slice(beg, end)
                     for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
    inner_bb = tuple(slice(beg, end)
                     for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
    local_bb = tuple(slice(beg, end)
                     for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))

    # load the mask with halo if necessary
    mask = ds_mask[outer_bb].astype('bool')

    # don't process empty blocks
    if np.sum(mask[local_bb]) == 0:
        with open(res_file, 'w') as f:
            json.dump({'t': time.time() - t0}, f)
        return

    # if this is the second pass, load the fragments
    # we already have as seeds
    if second_pass:
        initial_seeds = ds_out[outer_bb]
    else:
        initial_seeds = None

    # load affinities and make heightmap for the watershed
    bb_affs = (slice(1, 3),) + outer_bb
    affs = ds_affs[bb_affs]
    if affs.dtype == np.dtype('uint8'):
        affs = affs.astype('float32') / 255.
    affs = np.mean(1. - affs, axis=0)

    # load the mask and make the invalid mask by inversion
    inv_mask = np.logical_not(mask)
    affs[inv_mask] = 1

    # get the maxima seeds on 2d distance transform to fill gaps
    # in the extended seeds
    seeds = compute_max_seeds(affs, boundary_threshold, sigma_maxima, offset, inv_mask, initial_seeds)

    # run the watershed
    seeds, max_id = run_2d_ws(affs, seeds, inv_mask, size_filter)

    # write the result
    ds_out[inner_bb] = seeds[local_bb]
    with open(res_file, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


def run_watersheds_2d(path, aff_key, mask_key, out_key,
                      job_id, config_file, tmp_folder):

    f5 = z5py.File(path)
    ds_affs = f5[aff_key]
    ds_mask = f5[mask_key]
    ds_out = f5[out_key]

    with open(config_file) as f:
        input_config = json.load(f)
        block_list = input_config['block_list']
        config = input_config['config']
        block_shape = config['block_shape']

    shape = ds_mask.shape
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))

    [ws_block(ds_affs, ds_mask, ds_out,
              blocking, int(block_id), config,
              tmp_folder) for block_id in block_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('aff_key', type=str)
    parser.add_argument('mask_key', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('job_id', type=int)
    parser.add_argument('config_file', type=str)
    parser.add_argument('tmp_folder', type=str)
    args = parser.parse_args()

    run_watersheds_2d(args.path,
                      args.aff_key,
                      args.mask_key,
                      args.out_key,
                      args.job_id,
                      args.config_file,
                      args.tmp_folder)
