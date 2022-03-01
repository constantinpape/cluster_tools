#! /bin/python

import os
import sys
import json

import luigi
import dask
import numpy as np
import toolz as tz
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.inference.frameworks import get_predictor, get_preprocessor
from cluster_tools.inference.prep_model import get_prep_model


#
# Inference Tasks
#


class InferenceBase(luigi.Task):
    """ Inference base class
    """

    task_name = "inference"
    src_file = os.path.abspath(__file__)

    # input volume, output volume and inference parameter
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.DictParameter()
    checkpoint_path = luigi.Parameter()
    halo = luigi.ListParameter()
    mask_path = luigi.Parameter(default="")
    mask_key = luigi.Parameter(default="")
    framework = luigi.Parameter(default="pytorch")
    #
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"dtype": "uint8", "compression": "gzip", "chunks": None,
                       "device_mapping": None, "use_best": True, "tda_config": {},
                       "prep_model": None, "gpu_type": "gpu=2080Ti",
                       "channel_accumulation": None, "mixed_precision": False,
                       "preprocess_kwargs": {}})
        return config

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        assert self.framework in ("pytorch", "inferno", "bioimageio")

        # get the global config and init configs
        (shebang, block_shape,
         roi_begin, roi_end,
         block_list_path) = self.global_config_values(with_block_list_path=True)
        self.init(shebang)

        # load the task config
        config = self.get_task_config()
        dtype = config.pop("dtype", "uint8")
        compression = config.pop("compression", "gzip")
        chunks = config.pop("chunks", None)
        assert dtype in ("uint8", "float32")

        # get shapes and chunks
        shape = vu.get_shape(self.input_path, self.input_key)
        chunks = tuple(chunks) if chunks is not None else tuple(bs // 2 for bs in block_shape)
        # make sure block shape can be divided by chunks
        assert all(bs % ch == 0 for ch, bs in zip(chunks, block_shape)),\
            "%s, %s" % (str(chunks), block_shape)

        # check if we have single dataset or multi dataset output
        out_key_dict = self.output_key
        output_keys = list(out_key_dict.keys())
        channel_mapping = list(out_key_dict.values())

        channel_accumulation = config.get("channel_accumulation", None)

        # make output volumes
        with vu.file_reader(self.output_path) as f:
            for out_key, out_channels in zip(output_keys, channel_mapping):
                assert len(out_channels) == 2
                n_channels = out_channels[1] - out_channels[0]
                assert n_channels > 0
                if n_channels > 1 and channel_accumulation is None:
                    out_shape = (n_channels,) + shape
                    out_chunks = (1,) + chunks
                else:
                    out_shape = shape
                    out_chunks = chunks

                f.require_dataset(out_key, shape=out_shape,
                                  chunks=out_chunks, dtype=dtype, compression=compression)

        # update the config
        config.update({"input_path": self.input_path, "input_key": self.input_key,
                       "output_path": self.output_path, "checkpoint_path": self.checkpoint_path,
                       "block_shape": block_shape, "halo": self.halo,
                       "output_keys": output_keys, "channel_mapping": channel_mapping,
                       "framework": self.framework})
        if self.mask_path != "":
            assert self.mask_key != ""
            config.update({"mask_path": self.mask_path, "mask_key": self.mask_key})

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end,
                                             block_list_path=block_list_path)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class InferenceLocal(InferenceBase, LocalTask):
    """ Inference on local machine
    """
    pass


class InferenceSlurm(InferenceBase, SlurmTask):
    """ Inference on slurm cluster
    """
    # update the configs to run on slurm gpu nodes
    @staticmethod
    def default_task_config():
        conf = InferenceBase.default_task_config()
        slurm_requirements = conf.get("slurm_requirements", [])
        slurm_requirements.append(conf.get("gpu_type", "gpu=2080Ti"))
        conf.update({"slurm_requirements": slurm_requirements})
        return conf

    @staticmethod
    def default_global_config():
        conf = SlurmTask.default_global_config()
        conf.update({"partition": "gpu"})
        return conf


class InferenceLSF(InferenceBase, LSFTask):
    """ Inference on lsf cluster
    """
    pass


#
# Implementation
#


def _load_input(ds, offset, block_shape, halo, padding_mode="reflect"):

    shape = ds.shape
    starts = [off - ha for off, ha in zip(offset, halo)]
    stops = [off + bs + ha for off, bs, ha in zip(offset, block_shape, halo)]

    # we pad the input volume if necessary
    pad_left = None
    pad_right = None

    # check for padding to the left
    if any(start < 0 for start in starts):
        pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
        starts = [max(0, start) for start in starts]

    # check for padding to the right
    if any(stop > shape[i] for i, stop in enumerate(stops)):
        pad_right = tuple(stop - shape[i] if stop > shape[i] else 0 for i, stop in enumerate(stops))
        stops = [min(shape[i], stop) for i, stop in enumerate(stops)]

    bb = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    data = ds[bb]

    # pad if necessary
    if pad_left is not None or pad_right is not None:
        pad_left = (0, 0, 0) if pad_left is None else pad_left
        pad_right = (0, 0, 0) if pad_right is None else pad_right
        pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
        data = np.pad(data, pad_width, mode=padding_mode)

    return data


def _to_uint8(data, float_range=(0., 1.), safe_scale=True):
    if safe_scale:
        mult = np.floor(255./(float_range[1]-float_range[0]))
    else:
        mult = np.ceil(255./(float_range[1]-float_range[0]))
    add = 255 - mult*float_range[1]
    return np.clip((data*mult+add).round(), 0, 255).astype("uint8")


def _run_inference(blocking, block_list, halo, ds_in, ds_out, mask,
                   preprocess, predict, channel_mapping, channel_accumulation,
                   n_threads):

    block_shape = blocking.blockShape
    dtypes = [dso.dtype for dso in ds_out]
    dtype = dtypes[0]
    assert all(dtp == dtype for dtp in dtypes)

    @dask.delayed
    def log1(block_id):
        fu.log("start processing block %i" % block_id)
        return block_id

    @dask.delayed
    def load_input(block_id):
        block = blocking.getBlock(block_id)

        # if we have a mask, check if this block is in mask
        if mask is not None:
            bb = vu.block_to_bb(block)
            bb_mask = mask[bb].astype("bool")
            if np.sum(bb_mask) == 0:
                return block_id, None

        return block_id, _load_input(ds_in, block.begin, block_shape, halo)

    @dask.delayed
    def preprocess_impl(inputs):
        block_id, data = inputs
        if data is None:
            return block_id, None
        data = preprocess(data)
        return block_id, data

    @dask.delayed
    def predict_impl(inputs):
        block_id, data = inputs
        if data is None:
            return block_id, None
        data = predict(data)
        return block_id, data

    # TODO de-spagehttify
    @dask.delayed
    def write_output(inputs):
        block_id, output = inputs

        if output is None:
            return block_id

        out_shape = output.shape
        if len(out_shape) == 3:
            assert len(ds_out) == 1
        bb = vu.block_to_bb(blocking.getBlock(block_id))

        # check if we need to crop the output
        # NOTE this is not cropping the halo, which is done beforehand in the
        # predictor already, but to crop overhanging chunks at the end of th dataset
        actual_shape = [b.stop - b.start for b in bb]
        if actual_shape != block_shape:
            block_bb = tuple(slice(0, ash) for ash in actual_shape)
            if output.ndim == 4:
                block_bb = (slice(None),) + block_bb
            output = output[block_bb]

        # write the output to our output dataset(s)
        for dso, chann_mapping in zip(ds_out, channel_mapping):
            chan_start, chan_stop = chann_mapping

            if dso.ndim == 3:
                if channel_accumulation is None:
                    assert chan_stop - chan_start == 1
                out_bb = bb
            else:
                assert output.ndim == 4
                assert chan_stop - chan_start == dso.shape[0]
                out_bb = (slice(None),) + bb

            if output.ndim == 4:
                channel_output = output[chan_start:chan_stop].squeeze()
            else:
                channel_output = output

            # apply channel accumulation if specified
            if channel_accumulation is not None and channel_output.ndim == 4:
                channel_output = channel_accumulation(channel_output, axis=0)

            # cast to uint8 if necessary
            if dtype == "uint8":
                channel_output = _to_uint8(channel_output)

            dso[out_bb] = channel_output

        return block_id

    @dask.delayed
    def log2(block_id):
        fu.log_block_success(block_id)
        return 1

    # iterate over the blocks in block list, get the input data and predict
    results = []
    for block_id in block_list:
        res = tz.pipe(block_id, log1, load_input,
                      preprocess_impl, predict_impl,
                      write_output, log2)
        results.append(res)

    fu.log(f"Starting inference with {n_threads} threads")
    success = dask.compute(*results, scheduler="threads", num_workers=n_threads)
    fu.log("Finished prediction for %i blocks" % sum(success))


def inference(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    input_path = config["input_path"]
    input_key = config["input_key"]
    output_path = config["output_path"]
    checkpoint_path = config["checkpoint_path"]
    block_shape = config["block_shape"]
    block_list = config["block_list"]
    halo = config["halo"]
    framework = config["framework"]
    n_threads = config["threads_per_job"]
    use_best = config.get("use_best", True)
    mixed_precision = config.get("mixed_precision", False)
    channel_accumulation = config.get("channel_accumulation", None)
    if channel_accumulation is not None:
        fu.log("Accumulating channels with %s" % channel_accumulation)
        channel_accumulation = getattr(np, channel_accumulation)

    fu.log("run inference with framework %s, with %i threads" % (framework, n_threads))
    fu.log("input block size is %s and halo is %s" % (str(block_shape), str(halo)))

    output_keys = config["output_keys"]
    channel_mapping = config["channel_mapping"]

    device_mapping = config.get("device_mapping", None)
    if device_mapping is not None:
        device_id = device_mapping[str(job_id)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        fu.log("setting cuda visible devices to %i" % device_id)
    gpu = 0

    tda_config = config.get("tda_config", {})
    if tda_config:
        fu.log("Using test-time-data-augmentation with config:")
        fu.log(str(tda_config))

    fu.log("Loading model from %s" % checkpoint_path)

    prep_model = config.get("prep_model", None)
    if prep_model is not None:
        prep_model = get_prep_model(prep_model)

    predict = get_predictor(framework)(checkpoint_path, halo, gpu=gpu, prep_model=prep_model,
                                       use_best=use_best, mixed_precision=mixed_precision,
                                       **tda_config)
    fu.log("Have model")
    preprocess_kwargs = config.get("preprocess_kwargs", {})
    preprocess = get_preprocessor(framework, **preprocess_kwargs)

    shape = vu.get_shape(input_path, input_key)
    blocking = nt.blocking(roiBegin=[0, 0, 0],
                           roiEnd=list(shape),
                           blockShape=list(block_shape))

    with vu.file_reader(input_path, "r") as f_in, vu.file_reader(output_path, "a") as f_out:

        ds_in = f_in[input_key]
        ds_out = [f_out[key] for key in output_keys]

        if "mask_path" in config:
            mask_path, mask_key = config["mask_path"], config["mask_key"]
            fu.log("Load mask from %s:%s" % (mask_path, mask_key))
            mask = vu.load_mask(mask_path, mask_key, shape)
            fu.log("Have loaded mask")
        else:
            mask = None
        _run_inference(blocking, block_list, halo, ds_in, ds_out, mask,
                       preprocess, predict, channel_mapping,
                       channel_accumulation, n_threads)
    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    inference(job_id, path)
