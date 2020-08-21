#! /usr/bin/python

import os
import json
import sys
from glob import glob
from subprocess import check_output, CalledProcessError

import luigi
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.volume_utils as vu
from cluster_tools.utils.task_utils import DummyTask
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


class TransformixBase(luigi.Task):
    """ Transformix base class
    """
    formats = ('bdv', 'tif')

    # what about cubic etc?
    interpolation_modes = {'linear': 'FinalLinearInterpolator',
                           'nearest': 'FinalNearestNeighborInterpolator'}
    result_types = ('unsigned char', 'unsigned short')

    task_name = 'transformix'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path_file = luigi.Parameter()
    output_path_file = luigi.Parameter()
    transformation_file = luigi.Parameter()
    fiji_executable = luigi.Parameter()
    elastix_directory = luigi.Parameter()
    shape = luigi.ListParameter(default=None)
    resolution = luigi.ListParameter(default=None)
    interpolation = luigi.Parameter(default='nearest')
    output_format = luigi.Parameter(default='bdv')
    dependency = luigi.TaskParameter(default=DummyTask())

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        config.update({'ResultImagePixelType': None})
        return config

    # update the transformation with our interpolation mode
    # and the corresponding dtype
    def update_transformation(self, in_file, out_file, res_type):

        interpolator_name = self.interpolation_modes[self.interpolation]

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
                # change the interpolator
                if line.startswith("(ResampleInterpolator"):
                    line = update_line(line, interpolator_name, False)
                # change the pixel result type
                elif line.startswith("(ResultImagePixelType") and res_type is not None:
                    line = update_line(line, res_type, False)
                elif line.startswith("(Size") and self.shape is not None:
                    shape_str = " ".join(map(str, self.shape[::-1]))
                    line = update_line(line, shape_str, True)
                elif line.startswith("(Spacing") and self.resolution is not None:
                    resolution_str = " ".join(map(str, self.resolution[::-1]))
                    line = update_line(line, resolution_str, True)
                f_out.write(line)

    def update_transformations(self, res_type):
        trafo_folder, trafo_name = os.path.split(self.transformation_file)
        trafo_files = glob(os.path.join(trafo_folder, '*.txt'))

        out_folder = os.path.join(self.tmp_folder, 'transformations')
        os.makedirs(out_folder, exist_ok=True)

        for trafo in trafo_files:
            name = os.path.split(trafo)[1]
            out = os.path.join(out_folder, name)
            self.update_transformation(trafo, out, res_type)

        new_trafo = os.path.join(out_folder, trafo_name)
        assert os.path.exists(new_trafo)
        return new_trafo

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        with open(self.input_path_file) as f:
            inputs = json.load(f)
        with open(self.output_path_file) as f:
            outputs = json.load(f)

        assert len(inputs) == len(outputs), "%i, %i" % (len(inputs), len(outputs))
        assert all(os.path.exists(inp) for inp in inputs), f"{inputs}"
        n_files = len(inputs)

        assert os.path.exists(self.transformation_file)
        assert os.path.exists(self.fiji_executable)
        assert os.path.exists(self.elastix_directory)
        assert self.output_format in self.formats
        assert self.interpolation in self.interpolation_modes

        config = self.get_task_config()
        res_type = config.pop('ResultImagePixelType', None)
        if res_type is not None:
            assert res_type in self.result_types
        trafo_file = self.update_transformations(res_type)

        # get the split of file-ids to the volume
        file_list = vu.blocks_in_volume((n_files,), (1,))

        # we don't need any additional config besides the paths
        config.update({"input_path_file": self.input_path_file,
                       "output_path_file": self.output_path_file,
                       "transformation_file": trafo_file,
                       "fiji_executable": self.fiji_executable,
                       "elastix_directory": self.elastix_directory,
                       "tmp_folder": self.tmp_folder,
                       "output_format": self.output_format})

        # prime and run the jobs
        n_jobs = min(self.max_jobs, n_files)
        self.prepare_jobs(n_jobs, file_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class TransformixLocal(TransformixBase, LocalTask):
    """
    Transformix on local machine
    """
    pass


class TransformixSlurm(TransformixBase, SlurmTask):
    """
    Transformix on slurm cluster
    """
    pass


class TransformixLSF(TransformixBase, LSFTask):
    """
    Transformix on lsf cluster
    """
    pass


#
# Implementation
#

def apply_for_file(input_path, output_path,
                   transformation_file, fiji_executable,
                   elastix_directory, tmp_folder, n_threads,
                   output_format):

    assert os.path.exists(elastix_directory)
    assert os.path.exists(tmp_folder)
    assert os.path.exists(input_path)
    assert os.path.exists(transformation_file)

    if output_format == 'tif':
        format_str = 'Save as Tiff'
    elif output_format == 'bdv':
        format_str = 'Save as BigDataViewer .xml/.h5'
    else:
        assert False, "Invalid output format %s" % output_format

    trafo_dir, trafo_name = os.path.split(transformation_file)
    # transformix arguments need to be passed as one string,
    # with individual arguments comma separated
    # the argument to transformaix needs to be one large comma separated string
    transformix_argument = ["elastixDirectory=\'%s\'" % elastix_directory,
                            "workingDirectory=\'%s\'" % os.path.abspath(tmp_folder),
                            "inputImageFile=\'%s\'" % os.path.abspath(input_path),
                            "transformationFile=\'%s\'" % trafo_name,
                            "outputFile=\'%s\'" % os.path.abspath(output_path),
                            "outputModality=\'%s\'" % format_str,
                            "numThreads=\'%i\'" % n_threads]
    transformix_argument = ",".join(transformix_argument)
    transformix_argument = "\"%s\"" % transformix_argument

    # command based on https://github.com/embl-cba/fiji-plugin-elastixWrapper/issues/2:
    # srun --mem 16000 -n 1 -N 1 -c 8 -t 30:00 -o $OUT -e $ERR
    # /g/almf/software/Fiji.app/ImageJ-linux64  --ij2 --headless --run "Transformix"
    # "elastixDirectory='/g/almf/software/elastix_v4.8', workingDirectory='$TMPDIR',
    # inputImageFile='$INPUT_IMAGE',transformationFile='/g/cba/exchange/platy-trafos/linear/TransformParameters.BSpline10-3Channels.0.txt
    # outputFile='$OUTPUT_IMAGE',outputModality='Save as BigDataViewer .xml/.h5',numThreads='1'"
    cmd = [fiji_executable, "--ij2", "--headless", "--run", "\"Transformix\"", transformix_argument]

    cmd_str = " ".join(cmd)
    fu.log("Calling the following command:")
    fu.log(cmd_str)

    cwd = os.getcwd()
    try:
        # we need to change the working dir to the transformation directroy, so that relative paths in
        # the transformations are correct
        trafo_dir = os.path.split(transformation_file)[0]
        fu.log("Change directory to %s" % trafo_dir)
        os.chdir(trafo_dir)

        # check_output(cmd)
        # the CLI parser is very awkward.
        # I could only get it to work by passing the whole command string
        # and setting shell to True.
        # otherwise, it would parse something wrong, and do nothing but
        # throwing a warning:
        # [WARNING] Ignoring invalid argument: --run
        check_output([cmd_str], shell=True)
    except CalledProcessError as e:
        raise RuntimeError(e.output.decode('utf-8'))
    finally:
        fu.log("Go back to cwd: %s" % cwd)
        os.chdir(cwd)

    if output_format == 'tif':
        expected_output = output_path + '-ch0.tif'
    elif output_format == 'bdv':
        expected_output = output_path + '.xml'

    # the elastix plugin has the nasty habit of failing without throwing a proper error code.
    # so we check here that we actually have the expected output. if not, something went wrong.
    assert os.path.exists(expected_output), "The output %s is not there." % expected_output


def transformix(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # read the config
    with open(config_path) as f:
        config = json.load(f)

    # get list of the input and output paths
    input_file = config['input_path_file']
    with open(input_file) as f:
        inputs = json.load(f)
    output_file = config['output_path_file']
    with open(output_file) as f:
        outputs = json.load(f)

    transformation_file = config['transformation_file']
    fiji_executable = config['fiji_executable']
    elastix_directory = config['elastix_directory']
    tmp_folder = config['tmp_folder']
    working_dir = os.path.join(tmp_folder, 'work_dir%i' % job_id)
    output_format = config['output_format']

    os.makedirs(working_dir, exist_ok=True)

    file_list = config['block_list']
    n_threads = config.get('threads_per_job', 1)

    fu.log("Applying registration with:")
    fu.log("transformation_file: %s" % transformation_file)
    fu.log("fiji_executable: %s" % fiji_executable)
    fu.log("elastix_directory: %s" % elastix_directory)

    for file_id in file_list:
        fu.log("start processing block %i" % file_id)

        infile = inputs[file_id]
        outfile = outputs[file_id]
        fu.log("Input: %s" % infile)
        fu.log("Output: %s" % outfile)
        apply_for_file(infile, outfile,
                       transformation_file, fiji_executable,
                       elastix_directory, working_dir, n_threads,
                       output_format)
        fu.log_block_success(file_id)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    transformix(job_id, path)
