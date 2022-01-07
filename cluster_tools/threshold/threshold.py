#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


class ThresholdBase(luigi.Task):
    """ Threshold base class
    """

    task_name = "threshold"
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    threshold = luigi.FloatParameter()
    threshold_mode = luigi.Parameter(default="greater")
    channel = luigi.Parameter(default=None)
    # task that is required before running this task
    dependency = luigi.TaskParameter(DummyTask())

    threshold_modes = ("greater", "less", "equal")

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"sigma_prefilter": 0})
        return config

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end, block_list_path\
            = self.global_config_values(with_block_list_path=True)
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        assert self.threshold_mode in self.threshold_modes
        config = self.get_task_config()
        config.update({"input_path": self.input_path,
                       "input_key": self.input_key,
                       "output_path": self.output_path,
                       "output_key": self.output_key,
                       "block_shape": block_shape,
                       "threshold": self.threshold,
                       "threshold_mode": self.threshold_mode})

        # get chunks
        chunks = config.pop("chunks", None)
        if chunks is None:
            chunks = tuple(bs // 2 for bs in block_shape)

        # check if we have a multi-channel volume and specify a channel
        # to apply the threshold to
        if self.channel is None:
            # if no channel is specified, we need 3d input
            assert len(shape) == 3, str(len(shape))
        else:
            # if channel is specified, we need 4d input
            assert isinstance(self.channel, (int, tuple, list))
            assert len(shape) == 4, str(len(shape))
            if isinstance(self.channel, int):
                assert shape[0] > self.channel, "%i, %i" % (shape[0], self.channel)
            else:
                assert all(isinstance(chan, int) for chan in self.channel)
                assert shape[0] > max(self.channel), "%i, %i" % (shape[0], max(self.channel))
            shape = shape[1:]
            config.update({"channel": self.channel})

        # clip chunks
        chunks = tuple(min(ch, sh) for ch, sh in zip(chunks, shape))

        # make output dataset
        compression = config.pop("compression", "gzip")
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key,  shape=shape, dtype="uint8",
                              compression=compression, chunks=chunks)

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end,
                                         block_list_path=block_list_path)
        n_jobs = min(len(block_list), self.max_jobs)

        # we only have a single job to find the labeling
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(n_jobs)


class ThresholdLocal(ThresholdBase, LocalTask):
    """
    Threshold on local machine
    """
    pass


class ThresholdSlurm(ThresholdBase, SlurmTask):
    """
    Threshold on slurm cluster
    """
    pass


class ThresholdLSF(ThresholdBase, LSFTask):
    """
    Threshold on lsf cluster
    """
    pass


def _threshold_block(block_id, blocking,
                     ds_in, ds_out, threshold,
                     threshold_mode, channel, sigma):
    fu.log("start processing block %i" % block_id)
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    bb = vu.block_to_bb(block)
    if channel is None:
        input_ = ds_in[bb]
    else:
        channel_ = [channel] if isinstance(channel, int) else channel
        in_shape = (len(channel_),) + tuple(b.stop - b.start for b in bb)
        input_ = np.zeros(in_shape, dtype=ds_in.dtype)
        for chan_id, chan in enumerate(channel_):
            bb_inp = (slice(chan, chan + 1),) + bb
            input_[chan_id] = ds_in[bb_inp].squeeze()
        input_ = np.mean(input_, axis=0)

    input_ = vu.normalize(input_)
    if sigma > 0:
        input_ = vu.apply_filter(input_, "gaussianSmoothing", sigma)
        input_ = vu.normalize(input_)

    if threshold_mode == "greater":
        input_ = input_ > threshold
    elif threshold_mode == "less":
        input_ = input_ < threshold
    elif threshold_mode == "equal":
        input_ = input_ == threshold
    else:
        raise RuntimeError("Thresholding Mode %s not supported" % threshold_mode)

    ds_out[bb] = input_.astype("uint8")
    fu.log_block_success(block_id)


def threshold(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, "r") as f:
        config = json.load(f)
    input_path = config["input_path"]
    input_key = config["input_key"]
    output_path = config["output_path"]
    output_key = config["output_key"]
    block_list = config["block_list"]
    block_shape = config["block_shape"]
    threshold = config["threshold"]
    threshold_mode = config["threshold_mode"]

    sigma = config.get("sigma_prefilter", 0)
    channel = config.get("channel", None)

    fu.log("Applying threshold %f with mode %s" % (threshold, threshold_mode))

    with vu.file_reader(input_path, "r") as f_in, vu.file_reader(output_path) as f_out:

        ds_in = f_in[input_key]
        ds_out = f_out[output_key]

        shape = ds_in.shape
        if channel is not None:
            shape = shape[1:]
        assert len(shape) == 3

        blocking = nt.blocking([0, 0, 0], list(shape), block_shape)

        [_threshold_block(block_id, blocking,
                          ds_in, ds_out, threshold,
                          threshold_mode, channel, sigma) for block_id in block_list]

    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    threshold(job_id, path)
