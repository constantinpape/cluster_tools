import os
import shutil
import stat
import json
import time
from concurrent import futures
from subprocess import call, check_output
from datetime import datetime

import luigi


class BaseClusterTask(luigi.Task):
    """
    Base class for a task to run on the cluster.

    Subclasses need to implement
    - `prepare_jobs`: write config files etc. for the cluster jobs
    - `submit_jobs`: submit jobs to the cluster
    - `wait_for_jobs`: wait for all running jobs
    - `run`: custom run function
    - TODO requires ??
    The `run` method executes the job and must use the pre-implemented API
    functions in the correct order, see example below.

    Example:
        def run(self):
            # initialize the tmp folder and log dirs
            self.init()  # must-call

            # read config file and get number of n_jobs
            config = read_config() # not pre-implemented
            n_jobs, block_list = jobs_and_blocks() # not pre-implemented

            # prepare the jobs
            self.prepare_jobs(n_jobs, block_list, **config)  # must-call

            # submit the jobs
            self.submit_jobs(n_jobs)  # must-call

            # wait for jobs to finish
            self.wait_for_jobs()  # must-call

            # check for job success
            self.check_jobs(n_jobs)  # must-call
    """
    # temporary folder for configurations etc
    tmp_folder = luigi.Parameter()
    # TODO these shouldn't be luigi parameter
    # name of the task
    task_name = luigi.Parameter()
    # path to the python executable
    shebang = luigi.Parameter()

    #
    # API
    #

    def init(self):
        """
        Init tmp dir and python scripts.

        Should be the first call in `run`.
        """
        self._make_dirs()
        self._write_script_file()

    def check_jobs(self, n_jobs):
        """
        Check for jobs that ran successfully
        """
        success_list = []
        for job_id in range(n_jobs):
            log_file = os.path.join(self.tmp_folder, 'logs', '%s_job%i.log' % (self.task_name,
                                                                               job_id))
            # woot, there is no native tail in python ???
            last_line = check_output(['tail', '-1', log_file])[:-1]
            # get rid of the datetime prefix
            msg = " ".join(last_line.split()[2:])
            if msg == "processed job %i" % job_id:
                success_list.append(job_id)

        if len(success_list) == n_jobs:
            self._write_log("%s finished successfully" % self.task_name)
        else:
            failed_jobs = set(range(n_jobs)) - set(success_list)
            self._write_log("%s failed for jobs:" % self.task_name)
            self._write_log("%s" % ', '.join(map(str, failed_jobs)))
            # rename log file due to fail
            shutil.move(self.output().path,
                        os.path.join(self.tmp_folder, self.task_name + '_failed.log'))

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, self.task_name + '.log'))

    #
    # Must implement API
    #

    def prepare_jobs(self, n_jobs, block_list, **config):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    def submit_jobs(self, n_jobs):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    def wait_for_jobs(self):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    #
    # Helper functions
    #

    # TODO log levels ?
    def _write_log(self, msg):
        log_file = self.output().path
        with open(log_file, 'a') as f:
            f.write('%s: %s' % (str(datetime.now(), msg)))

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
        self._write_log('created tmp-folder and log dirs @ %s' % self.tmp_folder)

    # TODO allow config for individual blocks
    def _write_job_config(self, n_jobs, block_list, **config):
        # write the configurations for all jobs to the tmp folder
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_list': block_jobs, **config}
            config_path = self._config_path(job_id)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)
        self._write_log('written config for %i jobs' % n_jobs)

    # copy the python script to the temp folder and replace the shebang
    def _write_script_file(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        src_file = os.path.join(file_dir, self.task_name + '.py')
        assert os.path.exists(file_name), file_name
        trgt_file = os.path.join(self.tmp_folder, self.task_name + '.py')
        shtuil.copy(src_file, trgt_file)
        self._replace_shebang(trgt_file, self.shebang)
        self._make_executable(trgt_file)
        self._write_log('copied python script from %s to %s' % (src_file, trgt_file))

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
        # TODO set the job-name so that we can parse the squeue output properly
        slurm_template = ("#!/bin/bash\n"
                          "#SBATCH -A groupname\n"
                          "#SBATCH -N 1\n"
                          "#SBATCH -n %i\n" % self.cores_per_job
                          "#SBATCH --mem %s\n" % self.mem_limit
                          "#SBATCH -t %s\n" % self.time_limit
                          "#SBATCH -o %s\n" % os.path.join(self.tmp_folder,
                                                           'logs',
                                                           '%s_job$1.log' % self.task_name)
                          "#SBATCH -e %s\n" % os.path.join(self.tmp_folder,
                                                           'error_logs',
                                                           '%s_job$1.err' % self.task_name)
                          "%s %s" % (trgt_file, config_tmpl))
        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % self.task_name)
        with open(script_path, 'w') as f:
            f.write(script_path)

    def prepare_jobs(self, n_jobs, block_list, **config):
        # write the job configs
        self._write_job_config(n_jobs, block_list, **config)
        # write the slurm script file
        self._write_slurm_file()

    def submit_jobs(self, n_jobs):
        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % self.task_name)
        for job_id in range(n_jobs):
            call(['sbatch', script_path, str(job_id)])

    # TODO
    def wait_for_jobs(self):
        pass


class LocalTask(BaseClusterTask):
    """
    Task for running tasks locally for debugging /
    test purposes
    """

    def prepare_jobs(self, n_jobs, block_list, **config):
        # write the job configs
        self._write_job_config(n_jobs, block_list, **config)

    def submit_jobs(self, n_jobs):
        script_path = os.path.join(self.tmp_folder, self.task_name + '.py')

        def submit(job_id):
            log_file = os.path.join(self.tmp_folder, 'logs',
                                    '%s_job%i.log' % (self.task_name, job_id))
            err_file = os.path.join(self.tmp_folder, 'error_logs',
                                    '%s_job%i.err' % (self.task_name, job_id))
            config_file = self._config_path(job_id)
            with open(log_file, 'w') as f_out, open(err_file, 'w') as f_err:
                call([script_path, config_file], stdout=f_out, stderr=f_err)

        with futures.ProcessPoolExecutor(n_jobs) as pp:
            tasks = [pp.submit(submit, job_id) for job_id in range(n_jobs)]
            [t.result() for t in tasks]

    # don't need to wait for process pool
    def wait_for_jobs(self):
        pass


class LSFTask(BaseClusterTask):
    """
    Task for cluster with LSF scheduling system
    (tested on Janelia cluster)
    """
    # number of cores per job
    cores_per_job = luigi.IntParameter(default=1)
    # time limit in minutes (TODO write proper parser)
    time_limit = luigi.StringParameter(default='60')

    def prepare_jobs(self, n_jobs, block_list, **config):
        # write the job configs
        self._write_job_config(n_jobs, block_list, **config)

    def submit_jobs(self, n_jobs):
        script_path = os.path.join(self.tmp_folder, self.task_name + '.py')
        assert os.path.exists(script_path), script_path

        for job_id in range(n_jobs):
            config_file = self._config_path(job_id)
            command = '%s %s' % (script_path, config_file)
            log_file = os.path.join(self.tmp_folder, 'logs',
                                    '%s_job%i.log' % (self.task_name, job_id))
            err_file = os.path.join(self.tmp_folder, 'error_logs',
                                    '%s_job%i.err' % (self.task_name, job_id))
            bsub_command = 'bsub -n %i -J %s_%i -We %s -o %s -e %s \'%s\'' % (self.cores_per_job,
                                                                              self.task_name,
                                                                              job_id,
                                                                              self.time_limit,
                                                                              log_file, err_file,
                                                                              command)
            call([bsub_command], shell=True)

    def wait_for_jobs(self):
        # TODO move to some config
        wait_time = 10
        max_wait_time = None
        t_start = time.time()
        while True:
            time.sleep(wait_time)
            # TODO filter for job name pattern
            n_running = subprocess.check_output(['bjobs | grep $USER | wc -l'],
                                                shell=True).decode()
            n_running = int(n_running.strip('\n'))
            if n_running == 0:
                break
            if max_wait_time is not None:
                t_wait = time.time() - t_start
                if t_wait > max_wait_time:
                    # TODO cancel jobs with pattern
                    print("MAX WAIT TIME EXCEEDED")
                    break
