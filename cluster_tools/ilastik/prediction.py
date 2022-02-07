#! /usr/bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

try:
    import lazyflow
    from ilastik.experimental.api import from_project_file
    # set the number of threads used by ilastik to 0.
    # otherwise it does not work inside of the torch loader (and we want to limit number of threads anyways)
    # see https://github.com/ilastik/ilastik/issues/2517
    lazyflow.request.Request.reset_thread_pool(0)
except ImportError:
    from_project_file = None
try:
    from xarray import DataArray
except ImportError:
    DataArray = None


class PredictionBase(luigi.Task):
    """ Prediction base class
    """

    task_name = "prediction"
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    mask_path = luigi.Parameter(default="")
    mask_key = luigi.Parameter(default="")
    ilastik_project = luigi.Parameter()
    halo = luigi.ListParameter()
    out_channels = luigi.ListParameter(default=None)

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"dtype": "float32"})
        return config

    # would be nice to get this directly from the ilastik project instead
    # of running inference once, see also
    # https://github.com/ilastik/ilastik/issues/2530
    def get_out_channels(self, input_shape, input_channels):
        ilp = from_project_file(self.ilastik_project)
        dims = ("z", "y", "x")
        if input_channels is not None:
            input_shape = (input_channels,) + input_shape
            dims = ("c",) + dims
        input_ = np.random.rand(*input_shape).astype("float32")
        input_ = DataArray(input_, dims=dims)
        pred = ilp.predict(input_)
        n_out_channes = pred.shape[-1]
        return list(range(n_out_channes))

    def run_impl(self):
        assert from_project_file is not None
        assert DataArray is not None
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        shape = vu.get_shape(self.input_path, self.input_key)
        if len(shape) == 4:
            in_channels = shape[0]
            shape = shape[1:]
        else:
            in_channels = None
        assert len(shape) == 3
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)

        out_channels = self.out_channels
        if out_channels is None:
            out_channels = self.get_out_channels(block_shape, in_channels)

        # create the output dataset
        chunks = tuple(bs // 2 for bs in block_shape)
        n_channels = len(out_channels)
        if n_channels > 1:
            shape = (n_channels,) + shape
            chunks = (1,) + chunks
        dtype = config.get("dtype", "float32")
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=shape, chunks=chunks, dtype=dtype, compression="gzip")

        # update the config with input and output paths and keys
        # as well as block shape
        config.update({"input_path": self.input_path, "input_key": self.input_key,
                       "output_path": self.output_path, "output_key": self.output_key,
                       "halo": self.halo, "ilastik_project": self.ilastik_project,
                       "out_channels": out_channels, "block_shape": block_shape})
        if self.mask_path != "":
            assert self.mask_key != ""
            config.update({"mask_path": self.mask_path, "mask_key": self.mask_key})

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class PredictionLocal(PredictionBase, LocalTask):
    """ Prediction on local machine
    """
    pass


class PredictionSlurm(PredictionBase, SlurmTask):
    """ Prediction on slurm cluster
    """
    pass


class PredictionLSF(PredictionBase, LSFTask):
    """ Prediction on lsf cluster
    """
    pass


# TODO implement more dtype conversion
def _to_dtype(input_, dtype):
    idtype = input_.dtype
    if np.dtype(dtype) == idtype:
        return input_
    elif dtype == "uint8":
        input_ *= 255.
        return input_.astype("uint8")
    else:
        raise NotImplementedError(dtype)


def _pad_if_necessary(data, shape):
    if data.shape == shape:
        return data, None
    pad_width = []
    crop = []
    for dsh, sh in zip(data.shape, shape):
        if dsh == sh:
            pad_width.append((0, 0))
            crop.append(slice(None))
        else:
            assert sh > dsh
            pad_width.append((0, sh - dsh))
            crop.append(slice(0, dsh))
    data = np.pad(data, pad_width)
    assert data.shape == shape
    return data, tuple(crop)


def _predict_block(block_id, blocking, ilp, ds_in, ds_out, ds_mask, halo, out_channels):
    fu.log("Start processing block %i" % block_id)
    block = blocking.getBlockWithHalo(block_id, halo)
    bb = vu.block_to_bb(block.outerBlock)

    # check if there is any data to be processed, if we have a mask
    if ds_mask is not None:
        bb_mask = ds_mask[bb].astype("bool")
        if np.sum(bb_mask) == 0:
            fu.log_block_success(block_id)
            return

    dims = ("z", "y", "x")
    if ds_in.ndim == 4:
        bb = (slice(None),) + bb
        dims = ("c",) + dims
    input_ = ds_in[bb]

    # we need to pad to the full size for border chunks, because otherwise some filters may not be valid
    full_block_shape = tuple(sh + 2*ha for sh, ha in zip(blocking.blockShape, halo))
    input_, crop = _pad_if_necessary(input_, full_block_shape)

    # if we have a mask should set it as prediction mask in ilastik
    # (currently not supported by ilastik)
    pred = ilp.predict(DataArray(input_, dims=dims)).values
    if crop is not None:
        pred = pred[crop]

    inner_bb = vu.block_to_bb(block.innerBlockLocal)
    inner_bb = inner_bb + (tuple(out_channels),)
    pred = pred[inner_bb]
    if pred.shape[-1] == 1:
        pred = pred[..., 0]
    else:
        pred = pred.transpose((3, 0, 1, 2))
    pred = _to_dtype(pred, ds_out.dtype)
    assert pred.ndim in (3, 4)

    bb = vu.block_to_bb(block.innerBlock)
    if pred.ndim == 4:
        bb = (slice(None),) + bb
    ds_out[bb] = pred
    fu.log_block_success(block_id)


def prediction(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path, "r") as f:
        config = json.load(f)

    input_path = config["input_path"]
    input_key = config["input_key"]
    halo = config["halo"]
    ilastik_project = config["ilastik_project"]

    output_path = config["output_path"]
    output_key = config["output_key"]
    out_channels = config["out_channels"]

    assert os.path.exists(ilastik_project), ilastik_project
    assert os.path.exists(input_path)

    block_shape = config["block_shape"]
    block_list = config["block_list"]
    shape = vu.get_shape(input_path, input_key)
    if len(shape) == 4:
        shape = shape[1:]
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    ilp = from_project_file(ilastik_project)
    fu.log("start ilastik prediction")
    with vu.file_reader(input_path, "r") as f_in, vu.file_reader(output_path, "a") as f_out:
        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        if "mask_path" in config:
            mask_path, mask_key = config["mask_path"], config["mask_key"]
            fu.log("Load mask from %s:%s" % (mask_path, mask_key))
            mask = vu.load_mask(mask_path, mask_key, shape)
            fu.log("Have loaded mask")
        else:
            mask = None

        for block_id in block_list:
            _predict_block(block_id, blocking, ilp, ds_in, ds_out, mask, halo, out_channels)

    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    prediction(job_id, path)
