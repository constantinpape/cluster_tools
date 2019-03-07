#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.segmentation_utils as su
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Block-wise mutex watershed tasks
#

# TODO add size-filter
class MwsBlocksBase(luigi.Task):
    """ MwsBlocks base class
    """

    task_name = 'mws_blocks'
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    offsets = luigi.ListParameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')
    halo = luigi.ListParameter()
    serialize_overlap = luigi.BoolParameter(default=False)
    dependency = luigi.TaskParameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'strides': [1, 1, 1], 'randomize_strides': False,
                       'size_filter': 25, 'noise_level': 0.})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def requires(self):
        return self.dependency

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
                       'tmp_folder': self.tmp_folder, 'halo': self.halo,
                       'serialize_overlap': self.serialize_overlap})

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

        # make folder for overlap results
        os.makedirs(os.path.join(self.tmp_folder, 'mws_overlaps'), exist_ok=True)

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class MwsBlocksLocal(MwsBlocksBase, LocalTask):
    """
    MwsBlocks on local machine
    """
    pass


class MwsBlocksSlurm(MwsBlocksBase, SlurmTask):
    """
    MwsBlocks on slurm cluster
    """
    pass


class MwsBlocksLSF(MwsBlocksBase, LSFTask):
    """
    MwsBlocks on lsf cluster
    """
    pass


def _serialize_overlap(seg, blocking, block_id, halo, tmp_folder):
    shape = seg.shape
    halo_left, halo_right = halo
    for _, face_a, face_b, _, ngb_id in vu.iterate_faces(blocking, block_id,
                                                         return_only_lower=False):
        assert ngb_id != block_id
        to_lower = ngb_id < block_id
        axis = vu.faces_to_ovlp_axis(face_a, face_b)
        ovlp_slice = slice(0, 2 * halo_left[axis]) if to_lower else\
            slice(shape[axis] - 2 * halo_right[axis], shape[axis])
        face = tuple(slice(hal, sh - har) if dim != axis else ovlp_slice
                     for dim, sh, hal, har in zip(range(3), shape, halo_left, halo_right))
        ovlp = seg[face]
        if np.sum(ovlp > 0) > 0:
            save_path = os.path.join(tmp_folder,
                                     'mws_overlaps/ovlp_%i_%i.npy' % (block_id, ngb_id))
            np.save(save_path, ovlp)


def _mws_block(block_id, blocking,
               ds_in, ds_out, offsets,
               strides, randomize_strides,
               halo, serialize_overlap,
               tmp_folder, noise_level):
    fu.log("start processing block %i" % block_id)

    block = blocking.getBlockWithHalo(block_id, halo)
    in_bb = vu.block_to_bb(block.outerBlock)

    aff_bb = (slice(None),) + in_bb
    affs = vu.normalize(ds_in[aff_bb])

    seg = su.mutex_watershed(affs, offsets, strides=strides,
                             randomize_strides=randomize_strides,
                             noise_level=noise_level)

    out_bb = vu.block_to_bb(block.innerBlock)
    local_bb = vu.block_to_bb(block.innerBlockLocal)
    ds_out[out_bb] = seg[local_bb]

    if serialize_overlap:
        # we need the actual halo here, which also might be different at both sides
        halo_left = tuple(ib - ob for ib, ob
                          in zip(block.innerBlock.begin, block.outerBlock.begin))
        halo_right = tuple(oe - ie for oe, ie
                           in zip(block.outerBlock.end, block.innerBlock.end))
        _serialize_overlap(seg, blocking, block_id, (halo_left, halo_right), tmp_folder)

    # log block success
    fu.log_block_success(block_id)
    return int(seg.max()) + 1


def _mws_block_with_mask(block_id, blocking,
                         ds_in, ds_out,
                         mask, offsets,
                         strides, randomize_strides,
                         halo, serialize_overlap,
                         tmp_folder, noise_level):
    fu.log("start processing block %i" % block_id)

    block = blocking.getBlockWithHalo(block_id, halo)
    in_bb = vu.block_to_bb(block.outerBlock)

    aff_bb = (slice(None),) + in_bb
    affs = vu.normalize(ds_in[aff_bb])
    bb_mask = mask[in_bb].astype('bool')

    seg = su.mutex_watershed(affs, offsets, strides=strides, mask=bb_mask,
                             randomize_strides=randomize_strides,
                             noise_level=noise_level)

    out_bb = vu.block_to_bb(block.innerBlock)
    local_bb = vu.block_to_bb(block.innerBlockLocal)
    ds_out[out_bb] = seg[local_bb]

    if serialize_overlap:
        # we need the actual halo here, which also might be different at both sides
        halo_left = tuple(ib - ob for ib, ob
                          in zip(block.innerBlock.begin, block.outerBlock.begin))
        halo_right = tuple(oe - ie for oe, ie
                           in zip(block.outerBlock.end, block.innerBlock.end))
        _serialize_overlap(seg, blocking, block_id, (halo_left, halo_right), tmp_folder)

    # log block success
    fu.log_block_success(block_id)
    return int(seg.max()) + 1


def mws_blocks(job_id, config_path):

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
    tmp_folder = config['tmp_folder']
    offsets = config['offsets']

    strides = config['strides']
    assert len(strides) == 3
    assert all(isinstance(stride, int) for stride in strides)
    randomize_strides = config['randomize_strides']
    assert isinstance(randomize_strides, bool)
    noise_level = config['noise_level']

    halo = config['halo']
    serialize_overlap = config['serialize_overlap']

    mask_path = config.get('mask_path', '')
    mask_key = config.get('mask_key', '')

    with vu.file_reader(input_path, 'r') as f_in, vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        shape = ds_in.shape[1:]
        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)

        if mask_path != '':
            mask = vu.load_mask(mask_path, mask_key, shape)
            id_offsets = [_mws_block_with_mask(block_id, blocking,
                                               ds_in, ds_out,
                                               mask, offsets,
                                               strides, randomize_strides,
                                               halo, serialize_overlap, tmp_folder,
                                               noise_level)
                          for block_id in block_list]

        else:
            id_offsets = [_mws_block(block_id, blocking,
                                     ds_in, ds_out,
                                     offsets, strides, randomize_strides,
                                     halo, serialize_overlap, tmp_folder,
                                     noise_level)
                          for block_id in block_list]

    offset_dict = {block_id: off for block_id, off in zip(block_list, id_offsets)}
    save_path = os.path.join(tmp_folder, 'mws_offsets_%i.json' % job_id)
    fu.log("saving mws offsets to %s" % save_path)
    with open(save_path, 'w') as f:
        json.dump(offset_dict, f)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    mws_blocks(job_id, path)
