#! /bin/python

import os
import sys
import json

import imageio
import luigi
from pybdv.downsample import get_downsampler, sample_shape

import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.volume_utils as vu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# copy tasks
#

class CopySourcesBase(luigi.Task):
    """ copy_sources base class
    """

    task_name = 'copy_sources'
    src_file = os.path.abspath(__file__)

    # input and output volumes
    input_files = luigi.ListParameter()
    output_files = luigi.ListParameter()
    metadata_format = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    chunks = luigi.ListParameter()
    key = luigi.Parameter(default=None)
    downscaling_mode = luigi.Parameter(default="nearest")
    metadata_dict = luigi.DictParameter(default={})
    names = luigi.Parameter(default=None)
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    def require_output_folders(self):
        output_folders = [os.path.split(out_file)[0] for out_file in self.output_files]
        output_folders = list(set(output_folders))
        for out_folder in output_folders:
            os.makedirs(out_folder, exist_ok=True)

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        self.require_output_folders()

        # load and update the task config
        task_config = self.get_task_config()
        task_config.update({"input_files": self.input_files, "output_files": self.output_files,
                            "metadata_format": self.metadata_format,
                            "key": self.key, "metadata_dict": {k: v for k, v in self.metadata_dict.items()},
                            "scale_factors": self.scale_factors, "chunks": self.chunks,
                            "names": self.names, "downscaling_mode": self.downscaling_mode})

        block_list = list(range(len(self.input_files)))
        self._write_log("scheduled %i blocks to run" % len(block_list))

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, task_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class CopySourcesLocal(CopySourcesBase, LocalTask):
    """
    copy_volume local machine
    """
    pass


class CopySourcesSlurm(CopySourcesBase, SlurmTask):
    """
    copy on slurm cluster
    """
    pass


class CopySourcesLSF(CopySourcesBase, LSFTask):
    """
    copy_volume on lsf cluster
    """
    pass


#
# Implementation
#


def load_source(input_file, key):
    if key is None:
        # image file that can be read with imageio
        source = imageio.imread(input_file)
    else:
        # image file that can be read with open_file
        with vu.file_reader(input_file, "r") as f:
            source = f[key][:]
    return source


def write_source(source, output_file,
                 scale_factors, chunks,
                 metadata_format, downscaling_mode):
    sampler = get_downsampler(downscaling_mode)
    kwargs = {"dimension_separator": "/"} if metadata_format == "ome.zarr" else {}
    with vu.file_reader(output_file, "a", **kwargs) as f:
        key = vu.get_format_key(metadata_format, scale=0)
        f.require_dataset(key, data=source, compression="gzip", chunks=tuple(chunks),
                          shape=source.shape, dtype=source.dtype)

        for scale, scale_factor in enumerate(scale_factors, 1):
            sampled_shape = sample_shape(source.shape, scale_factor)
            chunks = tuple(min(sh, ch) for sh, ch in zip(sampled_shape, chunks))
            source = sampler(source, scale_factor, sampled_shape)
            key = vu.get_format_key(metadata_format, scale=scale)
            f.require_dataset(key, data=source, compression="gzip", chunks=chunks,
                              shape=source.shape, dtype=source.dtype)


def copy_source(input_file, output_file, key,
                metadata_format, metadata_dict, name,
                scale_factors, chunks, downscaling_mode):
    source = load_source(input_file, key)
    write_source(source, output_file,
                 scale_factors, chunks,
                 metadata_format, downscaling_mode)
    metadata_dict.update({"setup_name": name})
    vu.write_format_metadata(metadata_format, output_file, metadata_dict, scale_factors)


def copy_sources(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    # read the input cofig
    input_files = config["input_files"]
    output_files = config["output_files"]
    key = config["key"]
    metadata_format = config["metadata_format"]
    metadata_dict = config["metadata_dict"]
    downscaling_mode = config["downscaling_mode"]
    scale_factors = config["scale_factors"]
    chunks = config["chunks"]
    names = config["names"]

    # these are the ids of files to copy in this job
    # the field is called block list because we are re-using functionality from 3d blocking logic
    file_ids = config["block_list"]

    for file_id in file_ids:
        name = None if names is None else names[file_id]
        copy_source(input_files[file_id], output_files[file_id], key,
                    metadata_format, metadata_dict, name,
                    scale_factors, chunks, downscaling_mode)

    # log success
    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    copy_sources(job_id, path)
