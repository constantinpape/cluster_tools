#! /bin/python

# IMPORTANT do threadctl import first (before numpy imports)
from threadpoolctl import threadpool_limits

import os
import sys
import json
from functools import partial
from concurrent import futures

import numpy as np

import luigi
import vigra
import nifty.tools as nt
from skimage.measure import block_reduce
from elf.util import downscale_shape as _downsample_shape

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# downscaling tasks
#


class DownscalingBase(luigi.Task):
    """ downscaling base class
    """

    task_name = "downscaling"
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # the scale used to downsample the data.
    # can be list to account for anisotropic downsacling
    scale_factor = luigi.Parameter()
    # scale prefix for unique task identifier
    scale_prefix = luigi.Parameter()
    halo = luigi.ListParameter(default=[])
    effective_scale_factor = luigi.ListParameter(default=[])
    dimension_separator = luigi.Parameter(default=None)
    dependency = luigi.TaskParameter(default=DummyTask())

    interpolatable_types = ("float32", "float64",
                            "uint8", "int8",
                            "uint16", "int16")

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"library": "vigra", "chunks": None, "compression": "gzip",
                       "library_kwargs": None})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def downsample_shape(self, shape):
        if len(shape) == 4:
            new_shape = (shape[0],) + _downsample_shape(shape[1:], self.scale_factor)
        else:
            new_shape = _downsample_shape(shape, self.scale_factor)
        return new_shape

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end, block_list_path = self.global_config_values(with_block_list_path=True)
        self.init(shebang)

        # get shape, dtype and make block config
        with vu.file_reader(self.input_path, mode="r") as f:
            prev_shape = f[self.input_key].shape
            dtype = f[self.input_key].dtype
        ndim = len(prev_shape)
        assert ndim in (2, 3, 4), f"Only support 2d, 3d or 4d inputs. Got {ndim}d"
        # we treat the first dimension as channel axis for 4d data
        effective_ndim = 3 if ndim == 4 else ndim

        shape = self.downsample_shape(prev_shape)
        self._write_log("downscaling with factor %s from shape %s to %s" % (str(self.scale_factor),
                                                                            str(prev_shape), str(shape)))

        # load the downscaling config
        task_config = self.get_task_config()

        # make sure that we have order 0 downscaling if our datatype is not interpolatable
        library = task_config.get("library", "vigra")
        assert library in ("vigra", "skimage"), "Downscaling is only supported with vigra or skimage"
        if dtype not in self.interpolatable_types:
            assert library == "vigra", "datatype %s is not interpolatable, set library to vigra" % dtype
            opts = task_config.get("library_kwargs", {})
            opts = {} if opts is None else opts
            order = opts.get("order", None)
            assert order == 0,\
                "datatype %s is not interpolatable, set 'library_kwargs' = {'order': 0} to downscale it" % dtype

        # get the scale factor and check if we
        # do isotropic scaling
        scale_factor = self.scale_factor
        if isinstance(scale_factor, int):
            pass
        elif all(sf == scale_factor[0] for sf in scale_factor):
            assert len(scale_factor) == effective_ndim
            scale_factor = scale_factor[0]
        else:
            assert len(scale_factor) == 3
            # for now, we only support downscaling in-plane if the scale-factor is anisotropic
            assert scale_factor[0] == 1
            assert scale_factor[1] == scale_factor[2]

        # read the output chunks
        chunks = task_config.pop("chunks", None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)
        else:
            chunks = tuple(chunks)
            # TODO verify chunks further
            assert len(chunks) == effective_ndim, f"Chunks must be {effective_ndim}d"
        chunks = tuple(min(ch, sh) for sh, ch in zip(shape, chunks))

        if ndim == 4:
            out_shape = (prev_shape[0],) + shape
            out_chunks = (1,) + chunks
        else:
            out_shape = shape
            out_chunks = chunks

        compression = task_config.pop("compression", "gzip")
        # require output dataset
        file_kwargs = {} if self.dimension_separator is None else dict(dimension_separator=self.dimension_separator)
        with vu.file_reader(self.output_path, mode="a", **file_kwargs) as f:
            f.require_dataset(self.output_key, shape=out_shape, chunks=out_chunks,
                              compression=compression, dtype=dtype)

        # update the config with input and output paths and keys
        # as well as block shape
        task_config.update({"input_path": self.input_path, "input_key": self.input_key,
                            "output_path": self.output_path, "output_key": self.output_key,
                            "block_shape": block_shape, "scale_factor": scale_factor,
                            "halo": self.halo if self.halo else None})

        # if we have a roi, we need to re-sample it
        if roi_begin is not None:
            assert roi_end is not None
            effective_scale = self.effective_scale_factor if\
                self.effective_scale_factor else scale_factor
            self._write_log("downscaling roi with effective scale %s" % str(effective_scale))
            self._write_log("ROI before scaling: %s to %s" % (str(roi_begin), str(roi_end)))
            if isinstance(effective_scale, int):
                roi_begin = [rb // effective_scale for rb in roi_begin]
                roi_end = [re // effective_scale if re is not None else sh
                           for re, sh in zip(roi_end, shape)]
            else:
                roi_begin = [rb // sf for rb, sf in zip(roi_begin, effective_scale)]
                roi_end = [re // sf if re is not None else sh
                           for re, sf, sh in zip(roi_end, effective_scale, shape)]
            self._write_log("ROI after scaling: %s to %s" % (str(roi_begin), str(roi_end)))

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end,
                                             block_list_path)
            self._write_log("scheduled %i blocks to run" % len(block_list))
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config, self.scale_prefix)
        self.submit_jobs(n_jobs, self.scale_prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.scale_prefix)
        self.check_jobs(n_jobs, self.scale_prefix)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + "_%s.log" % self.scale_prefix))


class DownscalingLocal(DownscalingBase, LocalTask):
    """
    downscaling on local machine
    """
    pass


class DownscalingSlurm(DownscalingBase, SlurmTask):
    """
    downscaling on slurm cluster
    """
    pass


class DownscalingLSF(DownscalingBase, LSFTask):
    """
    downscaling on lsf cluster
    """
    pass


#
# Implementation
#


@threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
def _ds_vol(x, out_shape, sampler, scale_factor, dtype):
    sample_2d = not isinstance(scale_factor, int)
    out = sampler(x, out_shape, sample_2d)
    if np.dtype(dtype) in (np.dtype('uint8'), np.dtype('uint16')):
        max_val = np.iinfo(np.dtype(dtype)).max
        np.clip(out, 0, max_val, out=out)
        np.round(out, out=out)
    return out.astype(dtype)


@threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
def _ds_block(blocking, block_id, ds_in, ds_out, scale_factor, halo, sampler):
    fu.log("start processing block %i" % block_id)

    # load the block (output dataset / downsampled) coordinates
    if halo is None:
        block = blocking.getBlock(block_id)
        local_bb = np.s_[:]
        in_bb = vu.block_to_bb(block)
        out_bb = vu.block_to_bb(block)
        out_shape = block.shape
    else:
        halo_ds = [ha // scale_factor for ha in halo] if isinstance(scale_factor, int) else\
            [ha // sf for sf, ha in zip(scale_factor, halo)]
        block = blocking.getBlockWithHalo(block_id, halo_ds)
        in_bb = vu.block_to_bb(block.outerBlock)
        out_bb = vu.block_to_bb(block.innerBlock)
        local_bb = vu.block_to_bb(block.innerBlockLocal)
        out_shape = block.outerBlock.shape

    # check if we have channels
    ndim = ds_in.ndim
    in_shape = ds_in.shape
    if ndim == 4:
        in_shape = in_shape[1:]

    # upsample the input bounding box
    if isinstance(scale_factor, int):
        in_bb = tuple(slice(ib.start * scale_factor, min(ib.stop * scale_factor, sh))
                      for ib, sh in zip(in_bb, in_shape))
    else:
        in_bb = tuple(slice(ib.start * sf, min(ib.stop * sf, sh))
                      for ib, sf, sh in zip(in_bb, scale_factor, in_shape))
    # load the input
    if ndim == 4:
        in_bb = (slice(None),) + in_bb
        out_bb = (slice(None),) + out_bb
        local_bb = (slice(None),) + local_bb
    x = ds_in[in_bb]

    # don't sample empty blocks
    if np.sum(x != 0) == 0:
        fu.log_block_success(block_id)
        return

    dtype = x.dtype
    if np.dtype(dtype) != np.dtype("float32"):
        x = x.astype("float32")

    if ndim == 4:
        n_channels = x.shape[0]
        out = np.zeros((n_channels,) + tuple(out_shape), dtype=dtype)
        for c in range(n_channels):
            out[c] = _ds_vol(x[c], out_shape, sampler, scale_factor, dtype)
    else:
        out = _ds_vol(x, out_shape, sampler, scale_factor, dtype)

    try:
        ds_out[out_bb] = out[local_bb]
    except IndexError:
        raise(IndexError("%s, %s, %s" % (str(out_bb), str(local_bb), str(out.shape))))

    # log block success
    fu.log_block_success(block_id)


# wrap vigra.sampling.resize
@threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
def _ds_vigra(inp, output_shape, sample_2d, **vigra_kwargs):
    if sample_2d:
        out = np.zeros(output_shape, dtype="float32")
        for z in range(output_shape[0]):
            out[z] = vigra.sampling.resize(inp[z], shape=output_shape[1:], **vigra_kwargs)
        return out
    else:
        return vigra.sampling.resize(inp, output_shape, **vigra_kwargs)


# wrap skimage block_reduce
@threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
def _ds_skimage(inp, output_shape, sample_2d, block_size, func):
    return block_reduce(inp, block_size=block_size, func=func)


def _submit_blocks(ds_in, ds_out, block_shape, block_list,
                   scale_factor, halo, library,
                   library_kwargs, n_threads):

    # get the blocking
    ndim = ds_out.ndim
    shape = ds_out.shape
    if len(shape) == 4:
        ndim = 3
        shape = shape[1:]
    blocking = nt.blocking([0]*ndim, shape, block_shape)
    if library == "vigra":
        sampler = partial(_ds_vigra, **library_kwargs)
    elif library == "skimage":
        sk_scale = (scale_factor,) * ndim if isinstance(scale_factor, int) else tuple(scale_factor)
        ds_function = library_kwargs.get("function", "mean")
        ds_function = getattr(np, ds_function)
        sampler = partial(_ds_skimage, block_size=sk_scale, func=ds_function)
    else:
        raise ValueError("Invalid library %s, only vigra and skimage are supported" % library)

    if n_threads <= 1:
        for block_id in block_list:
            _ds_block(blocking, block_id, ds_in, ds_out,
                      scale_factor, halo, sampler)
    else:
        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_ds_block, blocking, block_id, ds_in, ds_out,
                               scale_factor, halo, sampler) for block_id in block_list]
            [t.result() for t in tasks]


def downscaling(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    # read the input cofig
    input_path = config["input_path"]
    input_key = config["input_key"]

    block_shape = list(config["block_shape"])
    block_list = config["block_list"]

    # read the output config
    output_path = config["output_path"]
    output_key = config["output_key"]

    scale_factor = config["scale_factor"]
    library = config.get("library", "vigra")
    library_kwargs = config.get("library_kwargs", None)
    if library_kwargs is None:
        library_kwargs = {}
    halo = config.get("halo", None)
    n_threads = config.get("threads_per_job", 1)

    # submit blocks
    # check if in and out - file are the same
    # because hdf5 does not like opening files twice
    if input_path == output_path:
        with vu.file_reader(output_path, mode="a") as f:
            ds_in = f[input_key]
            ds_out = f[output_key]
            _submit_blocks(ds_in, ds_out, block_shape, block_list, scale_factor, halo,
                           library, library_kwargs, n_threads)

    else:
        with vu.file_reader(input_path, mode="r") as f_in, vu.file_reader(output_path, mode="a") as f_out:
            ds_in = f_in[input_key]
            ds_out = f_out[output_key]
            _submit_blocks(ds_in, ds_out, block_shape, block_list, scale_factor, halo,
                           library, library_kwargs, n_threads)

    # log success
    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    downscaling(job_id, path)
