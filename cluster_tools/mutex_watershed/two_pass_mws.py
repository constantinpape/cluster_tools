#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt
import vigra
from elf.segmentation.mutex_watershed import mutex_watershed, mutex_watershed_with_seeds
from elf.segmentation.mutex_watershed import compute_grid_graph

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Block-wise mutex watershed tasks
#

# TODO add size-filter
class TwoPassMwsBase(luigi.Task):
    """ TwoPassMws base class
    """

    task_name = 'two_pass_mws'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    offsets = luigi.ListParameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')
    halo = luigi.ListParameter()
    dependency = luigi.TaskParameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'strides': [1, 1, 1], 'randomize_strides': False,
                       'size_filter': 25, 'noise_level': 0.})
        return config

    def requires(self):
        return self.dependency

    def _mws_pass(self, block_list, config, prefix):
        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config, prefix)
        self.submit_jobs(n_jobs, prefix)
        # wait till jobs finish and check for job success
        self.wait_for_jobs(prefix)
        self.check_jobs(n_jobs, prefix)

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)
        assert len(shape) == 4, "Need 4d input for MWS"
        n_channels = shape[0]
        shape = shape[1:]

        # TODO make optional which channels to choose
        assert len(self.offsets) == n_channels,\
            "%i, %i" % (len(self.offsets), n_channels)
        assert all(len(off) == 3 for off in self.offsets)

        config = self.get_task_config()
        config.update({'input_path': self.input_path, 'input_key': self.input_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'block_shape': block_shape, 'offsets': self.offsets,
                       'halo': self.halo, 'tmp_folder': self.tmp_folder})

        # check if we have a mask and add to the config if we do
        if self.mask_path != '':
            assert self.mask_key != ''
            config.update({'mask_path': self.mask_path, 'mask_key': self.mask_key})

        # get chunks
        chunks = config.pop('chunks', None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)
        # clip chunks
        chunks = tuple(min(ch, sh) for ch, sh in zip(chunks, shape))

        # make output dataset
        compression = config.pop('compression', 'gzip')
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key,  shape=shape, dtype='uint64',
                              compression=compression, chunks=chunks)

        blocking = nt.blocking([0, 0, 0], list(shape), list(block_shape))
        block_lists = vu.make_checkerboard_block_lists(blocking, roi_begin, roi_end)

        # we need the max-block-id to write out max-label-id later
        max_block_id = max([max(bl) for bl in block_lists])
        config.update({'max_block_id': max_block_id})

        for pass_id, block_list in enumerate(block_lists):
            config['pass'] = pass_id
            self._mws_pass(block_list, config, 'pass_%i' % pass_id)


class TwoPassMwsLocal(TwoPassMwsBase, LocalTask):
    """
    TwoPassMws on local machine
    """
    pass


class TwoPassMwsSlurm(TwoPassMwsBase, SlurmTask):
    """
    TwoPassMws on slurm cluster
    """
    pass


class TwoPassMwsLSF(TwoPassMwsBase, LSFTask):
    """
    TwoPassMws on lsf cluster
    """
    pass


def _write_nlabels(ds_out, seg):
    ds_out.attrs['maxId'] = int(seg.max())


def _mws_block_pass1(block_id, blocking,
                     ds_in, ds_out,
                     mask, offsets,
                     strides, randomize_strides,
                     halo, noise_level, max_block_id,
                     tmp_folder):
    fu.log("(Pass1) start processing block %i" % block_id)

    block = blocking.getBlockWithHalo(block_id, halo)
    in_bb = vu.block_to_bb(block.outerBlock)

    if mask is None:
        bb_mask = None
    else:
        bb_mask = mask[in_bb].astype('bool')
        if np.sum(bb_mask) == 0:
            fu.log_block_success(block_id)
            return

    aff_bb = (slice(None),) + in_bb
    affs = vu.normalize(ds_in[aff_bb])

    seg = mutex_watershed(affs, offsets, strides=strides, mask=bb_mask,
                          randomize_strides=randomize_strides,
                          noise_level=noise_level)

    out_bb = vu.block_to_bb(block.innerBlock)
    local_bb = vu.block_to_bb(block.innerBlockLocal)
    seg = seg[local_bb]

    # FIXME once vigra supports uint64 or we implement our own ...
    # seg = vigra.analysis.labelVolumeWithBackground(seg)

    # offset with lowest block coordinate
    offset_id = block_id * np.prod(blocking.blockShape)
    vigra.analysis.relabelConsecutive(seg, start_label=offset_id, keep_zeros=True, out=seg)
    ds_out[out_bb] = seg

    # get the state of the segmentation of this block
    grid_graph = compute_grid_graph(seg.shape, mask=bb_mask)
    affs = affs[(slice(None),) + local_bb]
    # FIXME this function yields incorrect uv-ids !
    state_uvs, state_weights, state_attractive = grid_graph.compute_state_for_segmentation(affs, seg, offsets,
                                                                                           n_attractive_channels=3,
                                                                                           ignore_label=True)
    # serialize the states
    save_path = os.path.join(tmp_folder, 'seg_state_block%i.h5' % block_id)
    with vu.file_reader(save_path) as f:
        f.create_dataset('edges', data=state_uvs)
        f.create_dataset('weights', data=state_weights)
        f.create_dataset('attractive_edge_mask', data=state_attractive)

    # write max-id for the last block
    if block_id == max_block_id:
        _write_nlabels(ds_out, seg)
    # log block success
    fu.log_block_success(block_id)


def _mws_block_pass2(block_id, blocking,
                     ds_in, ds_out,
                     mask, offsets,
                     strides, randomize_strides,
                     halo, noise_level, max_block_id,
                     tmp_folder):
    fu.log("(Pass2) start processing block %i" % block_id)

    block = blocking.getBlockWithHalo(block_id, halo)
    in_bb = vu.block_to_bb(block.outerBlock)

    if mask is None:
        # if we don't have a mask, initialize with fully 'in-mask' volume
        # bb_mask = np.ones(tuple(b.stop - b.begin for b in in_bb),
        #                   dtype='bool')
        bb_mask = None
    else:
        bb_mask = mask[in_bb].astype('bool')
        if np.sum(bb_mask) == 0:
            fu.log_block_success(block_id)
            return

    # TODO does this make sense ?
    # set the mask for parts of indirect neighbor blocks
    # (which are also part of pass 2) to 0
    # bb_mask = mask_corners(bb_mask, halo)

    aff_bb = (slice(None),) + in_bb
    affs = vu.normalize(ds_in[aff_bb])

    # load seeds
    seeds = ds_out[in_bb]
    seed_ids = np.unique(seeds)
    if seed_ids[0] == 0:
        seed_ids = seed_ids[1:]

    # load the serialized state for the neighboring (pass1) blocks
    # and find relevant edges between seed ids

    seed_edges = []
    seed_edge_weights = []
    attractive_mask = []

    for axis in range(3):
        for to_lower in (False, True):
            ngb_id = blocking.getNeighborId(block_id, axis, to_lower)

            # get path to state serialization and check if it exists
            save_path = os.path.join(tmp_folder, 'seg_state_block%i.h5' % ngb_id)
            if not os.path.exists(save_path):
                continue

            with vu.file_reader(save_path) as f:
                # first, load the edges and see if they have overlap with our seed ids
                ngb_edges = f['edges'][:]
                ngb_edge_mask = np.in1d(ngb_edges, seed_ids).reshape(ngb_edges.shape)
                ngb_edge_mask = ngb_edge_mask.all(axis=1)

                # if we have edges, load the corresponding weights
                # and attractive / repulsive state
                if ngb_edge_mask.sum() > 0:
                    ngb_edges = ngb_edges[ngb_edge_mask]
                    ngb_weights = f['weights'][:][ngb_edge_mask]
                    ngb_attractive_edges = f['attractive_edge_mask'][:][ngb_edge_mask]

                    seed_edges.append(ngb_edges)
                    seed_edge_weights.append(ngb_weights)
                    attractive_mask.append(ngb_attractive_edges)

    seed_edges = np.concatenate(seed_edges, axis=0)
    seed_edge_weights = np.concatenate(seed_edge_weights)
    attractive_mask = np.concatenate(attractive_mask)
    assert len(seed_edges) == len(seed_edge_weights) == len(attractive_mask)

    repulsive_mask = np.logical_not(attractive_mask)
    attractive_edges, repulsive_edges = seed_edges[attractive_mask], seed_edges[repulsive_mask]
    attractive_weights, repulsive_weights = seed_edge_weights[attractive_mask], seed_edge_weights[repulsive_mask]

    # run mws segmentation with seeds
    seed_state = {'attractive': (attractive_edges, attractive_weights),
                  'repulsive': (repulsive_edges, repulsive_weights)}
    seg, grid_graph = mutex_watershed_with_seeds(affs, offsets, seeds,
                                                 strides=strides, mask=bb_mask,
                                                 randomize_strides=randomize_strides,
                                                 noise_level=noise_level, return_graph=True,
                                                 seed_state=seed_state)
    # offset with lowest block coordinate
    offset_id = block_id * np.prod(blocking.blockShape)
    vigra.analysis.relabelConsecutive(seg, start_label=offset_id, keep_zeros=True, out=seg)

    # find assignment of seed ids to segmentation ids
    assignments = grid_graph.get_seed_assignments_from_node_labels(seg.flatten())

    # get the cropped segmentation
    local_bb = vu.block_to_bb(block.innerBlockLocal)
    seg_crop = seg[local_bb]

    # filter the assignments from ids that are not in the crop
    crop_ids = np.unique(seg_crop)
    filter_mask = np.in1d(assignments[:, 1], crop_ids)
    assignments = assignments[filter_mask]

    # store assignments to tmp folder
    save_path = os.path.join(tmp_folder, 'mws_two_pass_assignments_block_%i.npy' % block_id)
    np.save(save_path, assignments)

    out_bb = vu.block_to_bb(block.innerBlock)
    ds_out[out_bb] = seg_crop

    # write max-id for the last block
    if block_id == max_block_id:
        _write_nlabels(ds_out, seg)
    # log block success
    fu.log_block_success(block_id)


def two_pass_mws(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    block_shape = config['block_shape']
    block_list = config['block_list']
    offsets = config['offsets']

    strides = config['strides']
    assert len(strides) == 3
    assert all(isinstance(stride, int) for stride in strides)
    randomize_strides = config['randomize_strides']
    assert isinstance(randomize_strides, bool)
    noise_level = config['noise_level']

    halo = config['halo']

    mask_path = config.get('mask_path', '')
    mask_key = config.get('mask_key', '')
    pass_id = config['pass']
    tmp_folder = config['tmp_folder']
    max_block_id = config['max_block_id']

    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        shape = ds_in.shape[1:]
        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)

        if mask_path != '':
            mask = vu.load_mask(mask_path, mask_key, shape)
        else:
            mask = None

        mws_fu = _mws_block_pass1 if pass_id == 0 else _mws_block_pass2

        [mws_fu(block_id, blocking,
                ds_in, ds_out,
                mask, offsets,
                strides, randomize_strides,
                halo,  noise_level, max_block_id,
                tmp_folder)
         for block_id in block_list]

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    two_pass_mws(job_id, path)
