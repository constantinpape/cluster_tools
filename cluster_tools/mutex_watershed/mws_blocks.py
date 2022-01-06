#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt
import vigra
from elf.segmentation.mutex_watershed import mutex_watershed
from elf.segmentation.watershed import apply_size_filter

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Block-wise mutex watershed tasks
#

class MwsBlocksBase(luigi.Task):
    """ MwsBlocks base class
    """

    task_name = "mws_blocks"
    src_file = os.path.abspath(__file__)

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    offsets = luigi.ListParameter()
    mask_path = luigi.Parameter(default="")
    mask_key = luigi.Parameter(default="")
    halo = luigi.ListParameter(default=None)
    dependency = luigi.TaskParameter()

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"strides": [1, 1, 1], "randomize_strides": False,
                       "size_filter": 25, "noise_level": 0.})
        return config

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end, block_list_path\
            = self.global_config_values(with_block_list_path=True)
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)
        assert len(shape) == 4, "Need 4d input for MWS"
        n_channels = shape[0]
        shape = shape[1:]

        # TODO make optional which channels to choose
        assert len(self.offsets) == n_channels, "%i, %i" % (len(self.offsets), n_channels)
        assert all(len(off) == 3 for off in self.offsets)

        config = self.get_task_config()
        config.update({"input_path": self.input_path, "input_key": self.input_key,
                       "output_path": self.output_path, "output_key": self.output_key,
                       "block_shape": block_shape, "offsets": self.offsets, "halo": self.halo})

        # check if we have a mask and add to the config if we do
        if self.mask_path != "":
            assert self.mask_key != ""
            config.update({"mask_path": self.mask_path, "mask_key": self.mask_key})

        # get chunks
        chunks = config.pop("chunks", None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)
        # clip chunks
        chunks = tuple(min(ch, sh) for ch, sh in zip(chunks, shape))

        # make output dataset
        compression = config.pop("compression", "gzip")
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key,  shape=shape, dtype="uint64",
                              compression=compression, chunks=chunks)

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end,
                                         block_list_path=block_list_path)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)
        # wait till jobs finish and check for job success
        self.wait_for_jobs()
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


def _get_bbs(blocking, block_id, halo):
    if halo is None:
        block = blocking.getBlock(block_id)
        in_bb = out_bb = vu.block_to_bb(block)
        local_bb = np.s_[:]
    else:
        block = blocking.getBlockWithHalo(block_id, halo)
        in_bb = vu.block_to_bb(block.outerBlock)
        out_bb = vu.block_to_bb(block.innerBlock)
        local_bb = vu.block_to_bb(block.innerBlockLocal)
    return in_bb, out_bb, local_bb


def _mws_block(block_id, blocking,
               ds_in, ds_out,
               mask, offsets,
               strides, randomize_strides,
               halo, noise_level, size_filter):
    fu.log("start processing block %i" % block_id)

    in_bb, out_bb, local_bb = _get_bbs(blocking, block_id, halo)
    if mask is None:
        bb_mask = None
    else:
        bb_mask = mask[in_bb].astype("bool")
        if np.sum(bb_mask) == 0:
            fu.log_block_success(block_id)
            return

    aff_bb = (slice(None),) + in_bb
    affs = ds_in[aff_bb]
    if affs.sum() == 0:
        fu.log_block_success(block_id)
        return

    affs = vu.normalize(affs)
    seg = mutex_watershed(affs, offsets, strides=strides, mask=bb_mask,
                          randomize_strides=randomize_strides,
                          noise_level=noise_level)
    if size_filter > 0:
        hmap = np.max(affs[:(affs.ndim - 1)], axis=0)
        need_mask = bb_mask is not None and 0 in seg
        if need_mask:
            seg += 1
            exclude = [1]
        else:
            exclude = None
        seg, _ = apply_size_filter(seg.astype("uint32"), hmap, size_filter, exclude=exclude)
        if need_mask:
            seg[seg == 1] = 0
    seg = seg[local_bb].astype("uint64")

    # offset with lowest block coordinate
    offset_id = max(block_id * int(np.prod(blocking.blockShape)), 1)
    assert offset_id < np.iinfo("uint64").max, "Id overflow"
    vigra.analysis.relabelConsecutive(seg, start_label=offset_id, keep_zeros=True, out=seg)
    ds_out[out_bb] = seg

    # log block success
    fu.log_block_success(block_id)


def mws_blocks(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, "r") as f:
        config = json.load(f)
    input_path = config["input_path"]
    input_key = config["input_key"]
    output_path = config["output_path"]
    output_key = config["output_key"]
    block_shape = config["block_shape"]
    block_list = config["block_list"]
    offsets = config["offsets"]
    size_filter = config["size_filter"]

    strides = config["strides"]
    assert len(strides) == 3
    assert all(isinstance(stride, int) for stride in strides)
    randomize_strides = config["randomize_strides"]
    assert isinstance(randomize_strides, bool)
    noise_level = config["noise_level"]

    halo = config["halo"]

    mask_path = config.get("mask_path", "")
    mask_key = config.get("mask_key", "")

    with vu.file_reader(input_path, "r") as f_in, vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        shape = ds_in.shape[1:]
        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)
        mask = None if mask_path == "" else vu.load_mask(mask_path, mask_key, shape)

        [_mws_block(block_id, blocking,
                    ds_in, ds_out,
                    mask, offsets,
                    strides, randomize_strides,
                    halo, noise_level, size_filter) for block_id in block_list]
    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    mws_blocks(job_id, path)
