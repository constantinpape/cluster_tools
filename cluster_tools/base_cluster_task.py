import os
import shutil
import stat
import json
from concurrent import futures
from subprocess import call

import luigi


class BaseClusterTask(luigi.Task):
    """
    Base class for a task to run on the cluster.

    Subclasses need to implement
    - `prepare_jobs`: write config files etc. for the cluster jobs
    - `submit_jobs`: submit jobs to the cluster
    - `wait_for_jobs`: wait for all running jobs
    """
    # temporary folder for configurations etc
    tmp_folder = luigi.Parameter()
    # name of the task
    task_name = luigi.Parameter()
    # path to the python executable
    shebang = luigi.Parameter()

    def prepare_jobs(self, n_jobs, block_list, **config):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    def submit_jobs(self, n_jobs):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    # TODO args ?
    def wait_for_jobs(self):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    def _config_path(self, job_id):
        return os.path.join(self.tmp_folder, self.task_name + '_job%s.config' % str(job_id))

    # make the tmpdir and logdirs
    def _make_dirs(self):

        def mkdir(dirpath):
            try:
                os.mkdir(dirpath)
            except OSError:
                pass

        mkdir(self.tmp_folder)
        mkdir(os.path.join(self.tmp_folder, 'logs'))
        mkdir(os.path.join(self.tmp_folder, 'error_logs'))

    # TODO allow config for individual blocks
    def _write_job_config(self, n_jobs, block_list, **config):
        # write the configurations for all jobs to the tmp folder
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_list': block_jobs, **config}
            config_path = self._config_path(job_id)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    # copy the python script to the temp folder and replace the shebang
    def _write_script_file(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        src_file = os.path.join(file_dir, self.task_name + '.py')
        assert os.path.exists(file_name), file_name
        trgt_file = os.path.join(self.tmp_folder, self.task_name + '.py')
        shtuil.copy(src_file, trgt_file)
        self._replace_shebang(trgt_file, self.shebang)
        self._make_executable(trgt_file)

    # https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python
    @staticmethod
    def _replace_shebang(file_path, shebang):
        for i, line in enumerate(fileinput.input(file_path, inplace=True)):
            if i == 0:
                print(shebang, end='')
            else:
                print(line, end='')

    @staticmethod
    def _make_executable(path):
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)


class SlurmTask(BaseClusterTask):
    """
    Task for cluster with Slurm scheduling system
    (tested on EMBL cluster)
    """
    # number of cores per job
    cores_per_job = luigi.IntParameter(default=1)
    # memory limit (TODO write proper parser)
    mem_limit = luigi.StringParameter(default='1G')
    # time limit (TODO write proper parser)
    time_limit = luigi.StringParameter(default='0-1:00')

    def _write_slurm_file(self):
        trgt_file = os.path.join(self.tmp_folder, self.task_name + '.py')
        config_tmpl = self._config_path('$1')
        slurm_template = ("#!/bin/bash\n"
                          "#SBATCH -A groupname\n"
                          "#SBATCH -N 1\n"
                          "#SBATCH -n %i\n" % self.cores_per_job
                          "#SBATCH --mem %s\n" % self.mem_limit
                          "#SBATCH -t %s\n" % self.time_limit
                          "#SBATCH -o %s\n" % os.path.join(self.tmp_folder, 'logs', '%s_job$1.log' % self.task_name)
                          "#SBATCH -e %s\n" % os.path.join(self.tmp_folder, 'error_logs', '%s_job$1.err' % self.task_name)
                          "%s %s" % (trgt_file, config_tmpl))
        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % self.task_name)
        with open(script_path, 'w') as f:
            f.write(script_path)

    def prepare_jobs(self, n_jobs, block_list, **config):
        # make directories
        self._make_dirs()
        # write the job configs
        self._write_job_config(n_jobs, block_list, **config)
        # write the python script file
        self._write_script_file()
        # write the slurm script file
        self._write_slurm_file()

    def submit_jobs(self, n_jobs):
        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % self.task_name)
        for job_id in range(n_jobs):
            call(['sbatch', script_path, str(job_id)])

    def wait_for_jobs(self):
        pass


class LocalTask(BaseClusterTask):
    """
    Task for running tasks locally for debugging /
    test purposes
    """

    def prepare_jobs(self, n_jobs, block_list, **config):
        # make directories
        self._make_dirs()
        # write the job configs
        self._write_job_config(n_jobs, block_list, **config)
        # write the python script file
        self._write_script_file()

    def submit_jobs(self, n_jobs):
        script_path = os.path.join(self.tmp_folder, self.task_name + '.py')
        with futures.ProcessPoolExecutor(n_jobs) as pp:
            tasks = [pp.submit(call, [script_path, str(job_id)]) for job_id in range(n_jobs)]
            [t.result() for t in tasks]

    # don't need to wait for process pool
    def wait_for_jobs(self):
        pass

class LSFTask(BaseClusterTask):
    """
    Task for cluster with LSF scheduling system
    (tested on Janelia cluster)
    """
    pass
