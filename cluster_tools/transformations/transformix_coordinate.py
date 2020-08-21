#! /usr/bin/python

import os
import json
import subprocess
import sys
from glob import glob

import luigi
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.volume_utils as vu
import nifty.tools as nt

from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from elf.io import open_file
from nifty.transformation import coordinateTransformationZ5


class TransformixCoordinateBase(luigi.Task):
    """ TransformixCoordinate base class
    """
    task_name = 'transformix_coordinate'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    transformation_file = luigi.Parameter()
    elastix_directory = luigi.Parameter()

    shape = luigi.Parameter()
    resolution = luigi.Parameter(default=None)
    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'chunks': None, 'compression': 'gzip'})
        return config

    def requires(self):
        return self.dependency

    # update the transformation with our interpolation mode
    # and the corresponding dtype
    def update_transformation(self, in_file, out_file):

        def update_line(line, to_write, is_numeric):
            line = line.rstrip('\n')
            line = line.split()
            if is_numeric:
                line = [line[0], "%s)" % to_write]
            else:
                line = [line[0], "\"%s\")" % to_write]
            line = " ".join(line) + "\n"
            return line

        with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
            for line in f_in:
                if line.startswith("(Spacing") and self.resolution is not None:
                    resolution_str = " ".join(map(str, self.resolution[::-1]))
                    line = update_line(line, resolution_str, True)
                f_out.write(line)

    def update_transformations(self):
        trafo_folder, trafo_name = os.path.split(self.transformation_file)
        trafo_files = glob(os.path.join(trafo_folder, '*.txt'))

        out_folder = os.path.join(self.tmp_folder, 'transformations')
        os.makedirs(out_folder, exist_ok=True)

        for trafo in trafo_files:
            name = os.path.split(trafo)[1]
            out = os.path.join(out_folder, name)
            self.update_transformation(trafo, out)

        new_trafo = os.path.join(out_folder, trafo_name)
        assert os.path.exists(new_trafo)
        return new_trafo

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)
        config = self.get_task_config()

        with open_file(self.input_path, 'r') as f:
            dtype = f[self.input_key].dtype
        chunks = config['chunks']
        if chunks is None:
            chunks = block_shape
        compression = config['compression']

        with open_file(self.output_path, 'r') as f:
            f.require_dataset(self.output_key, shape=self.shape, chunks=chunks,
                              compression=compression, dtype=dtype)

        trafo_file = self.update_transformations()
        # we don't need any additional config besides the paths
        config.update({"input_path": self.input_path,
                       "input_key": self.input_key,
                       "output_path": self.output_path,
                       "output_key": self.output_key,
                       "transformation_file": trafo_file,
                       "elastix_directory": self.elastix_directory,
                       "tmp_folder": self.tmp_folder})

        block_list = vu.blocks_in_volume(self.shape, block_shape, roi_begin, roi_end)
        self._write_log("scheduled %i blocks to run" % len(block_list))

        # prime and run the jobs
        n_jobs = min(len(block_list), self.max_jobs)
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)

        # prime and run the jobs
        n_jobs = min(self.max_jobs, len(block_list))
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class TransformixCoordinateLocal(TransformixCoordinateBase, LocalTask):
    """
    TransformixCoordinate on local machine
    """
    pass


class TransformixCoordinateSlurm(TransformixCoordinateBase, SlurmTask):
    """
    TransformixCoordinate on slurm cluster
    """
    pass


class TransformixCoordinateLSF(TransformixCoordinateBase, LSFTask):
    """
    TransformixCoordinate on lsf cluster
    """
    pass


#
# Implementation
#


def _write_coords(starts, stops, out_file):
    n_coords = (stops[0] - starts[0]) * (stops[1] - stops[1]) * (stops[2] - starts[2])
    with open(out_file, 'w') as f:

        f.write("index\n")
        f.write(f"{n_coords}\n")

        for z in range(starts[0], stops[0]):
            for y in range(starts[1], stops[1]):
                for x in range(starts[2], stops[2]):
                    coord_str = " ".join(map(str, (x, y, z)))
                    f.write(f"{coord_str}\n")


def process_block(ds_in, ds_out,
                  blocking, block_id,
                  transformix_bin,
                  transformation_file,
                  tmp_folder):
    block = blocking.getBlock(block_id)
    coord_folder = os.path.join(tmp_folder, f'coords_{block_id}')

    # get the output coordinates for this block and write to file
    in_coord_file = os.path.join(coord_folder, 'inpoints.txt')
    _write_coords(block.begin, block.end, in_coord_file)

    # apply transformix to transform all coordinates
    cmd = [transformix_bin,
           '-def', in_coord_file,
           '-out', coord_folder,
           '-tp', transformation_file]
    subprocess.run(cmd)
    out_file = os.path.join(coord_folder, 'outputpoints.txt')

    # TODO support other data backends than n5/zarr
    # use nifty transformation functionality to apply the transformation
    bb = tuple(slice(beg, end)
               for beg, end in zip(block.begin, block.end))
    out = coordinateTransformationZ5(ds_in, out_file, bb)

    # write the output
    ds_out[bb] = out


def transformix_coordinate(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']

    output_path = config['output_path']
    output_key = config['output_key']

    transformation_file = config['transformation_file']
    elastix_dir = config['elastix_directory']
    tmp_folder = config['tmp_folder']

    block_list = config['block_list']
    block_shape = config['block_shape']

    fu.log("Applying registration with:")
    fu.log("transformation_file: %s" % transformation_file)
    fu.log("elastix_directory: %s" % elastix_dir)

    transformix_bin = os.path.join(elastix_dir, 'bin', 'transformix')
    # set the ld library path
    lib_path = os.environ['LD_LIBRARY_PATH']
    elastix_lib_path = os.path.join(elastix_dir, 'lib')
    os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{elastix_lib_path}"

    with open_file(input_path, 'r') as f_in, open_file(output_path, 'a') as f_out:

        ds_in = f_in[input_key]
        ds_out = f_out[output_key]
        shape = ds_out.shape

        blocking = nt.blocking([0, 0, 0], shape, block_shape)

        for block_id in block_list:
            fu.log("start processing block %i" % block_id)
            process_block(ds_in, ds_out,
                          blocking, block_id,
                          transformix_bin,
                          transformation_file,
                          tmp_folder)
            fu.log_block_success(block_id)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    transformix_coordinate(job_id, path)
