import os
import shutil
import stat
import json
import time
import fileinput
from concurrent import futures
from subprocess import call, check_output, CalledProcessError
from datetime import datetime, timedelta

import luigi

from .utils.function_utils import tail


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
    # maximum number of concurrent jobs
    max_jobs = luigi.IntParameter()
    # directory for configs
    config_dir = luigi.Parameter()

    #
    # API
    #

    def init(self, shebang):
        """ Init tmp dir and python scripts.

        Should be the first call in `run`.
        """
        self._write_script_file(shebang)

    def check_jobs(self, n_jobs, job_prefix=None):
        """ Check for jobs that ran successfully
        """
        success_list = []
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name, job_prefix)
        for job_id in range(n_jobs):
            log_file = os.path.join(self.tmp_folder, 'logs', '%s_%i.log' % (job_name, job_id))
            try:
                last_line = tail(log_file, 1)[0]
            # if the file does not exist, this throws a `CalledProcessError`
            # if it does exist, but is empty, it throws a `IndexError`
            # in bout cases, the job was unsuccessfull and we continue
            except (IndexError, CalledProcessError):
                continue
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
            raise RuntimeError("Task: %s failed for %i / %i jobs" % (self.task_name, len(failed_jobs), n_jobs))

    def get_task_config(self):
        """ Get the task configuration

        Reads the task config from 'config_dir/task_name.config'.
        If this does not exist, returns the default task config.
        """
        config_path = os.path.join(self.config_dir, self.task_name + '.config')
        if os.path.exists(config_path):
            self._write_log("reading task config from %s" % config_path)
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            self._write_log("reading default task config")
            return self.default_task_config()

    @staticmethod
    def default_task_config():
        """ Return the default task config

        The default implementation just contains `threads_per_job`, `time_limit` and `mem_limit`.
        Over-ride in deriving classes to specify the default configuration.
        """
        # time-limit in minutes
        # mem_limit in GB
        return {'threads_per_job': 1, 'time_limit': 60, 'mem_limit': 1.}

    def get_global_config(self):
        """ Get the global configuration

        Reads the global config from 'config_dir/global.config'.
        If this does not exist, returns the default global config.
        """
        config_path = os.path.join(self.config_dir, 'global.config')
        if os.path.exists(config_path):
            self._write_log("reading global config from %s" % config_path)
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            self._write_log("reading default global config")
            return self.default_global_config()

    @staticmethod
    def default_global_config():
        """ Return default global config
        """
        return {"block_shape": [50, 512, 512],
                "shebang": "#! /bin/python",
                "roi_begin": None,
                "roi_end": None,
                "groupname": "kreshuk"}

    def global_config_values(self):
        """ Load the global config values
        """
        config = self.get_global_config()
        # first two return values (shebang, block_shape) must exist
        # second two have defaults
        return (config["shebang"], config["block_shape"],
                config.get("roi_begin", None),
                config.get("roi_end", None))

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, self.task_name + '.log'))

    #
    # Must implement API
    #

    def prepare_jobs(self, n_jobs, block_list, config, job_prefix=None):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    def submit_jobs(self, n_jobs, job_prefix=None):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    def wait_for_jobs(self, job_prefix=None):
        raise NotImplementedError("BaseClusterTask does not implement any functionality")

    #
    # Helper functions
    #

    # TODO log levels ?
    def _write_log(self, msg):
        log_file = self.output().path
        with open(log_file, 'a') as f:
            f.write('%s: %s\n' % (str(datetime.now()), msg))

    def _config_path(self, job_id, job_prefix=None):
        if job_prefix is None:
            return os.path.join(self.tmp_folder, self.task_name + '_job_%s.config' % str(job_id))
        else:
            return os.path.join(self.tmp_folder, self.task_name + '_job_%s_%s.config' % (job_prefix, str(job_id)))

    # make the tmpdir and logdirs
    def make_dirs(self):

        def mkdir(dirpath):
            try:
                os.mkdir(dirpath)
            except OSError:
                pass

        mkdir(self.tmp_folder)
        mkdir(os.path.join(self.tmp_folder, 'logs'))
        mkdir(os.path.join(self.tmp_folder, 'error_logs'))
        self._write_log('created tmp-folder and log dirs @ %s' % self.tmp_folder)

    def _write_single_job_config(self, config, job_prefix):
        config_path = self._config_path(0, job_prefix)
        with open(config_path, 'w') as f:
            json.dump(config, f)

    def _write_multiple_job_configs(self, n_jobs, block_list, config, job_prefix):
        # write the configurations for all jobs to the tmp folder
        for job_id in range(n_jobs):
            block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_list': block_jobs, **config}
            config_path = self._config_path(job_id, job_prefix)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    # TODO allow config for individual blocks
    def _write_job_config(self, n_jobs, block_list, config, job_prefix=None):
        # check f we have a reduce style block, that is
        # not distributed over blocks
        if block_list is None:
            assert n_jobs == 1
            self._write_single_job_config(config, job_prefix)
        # otherwise, we have multiple jobs distributed over blocks
        else:
            self._write_multiple_job_configs(n_jobs, block_list, config, job_prefix)
        self._write_log('written config for %i jobs' % n_jobs)

    # copy the python script to the temp folder and replace the shebang
    def _write_script_file(self, shebang):
        assert os.path.exists(self.src_file), self.src_file
        trgt_file = os.path.join(self.tmp_folder, self.task_name + '.py')
        shutil.copy(self.src_file, trgt_file)
        self._replace_shebang(trgt_file, shebang)
        self._make_executable(trgt_file)
        self._write_log('copied python script from %s to %s' % (self.src_file, trgt_file))

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

    @staticmethod
    def _parse_time_limit(time_limit):
        """ Converts time limit in minutes to slurm format
        """
        tt = timedelta(minutes=time_limit) + datetime(1, 1, 1)
        return "%i-%i:%i:%i" % (tt.day - 1, tt.hour, tt.minute, tt.second)

    @staticmethod
    def _parse_mem_limit(mem_limit):
        """ Converts mem limit in GB to slurm format
        """
        if mem_limit > 1:
            # TODO I don't know if float mem
            return "%iG" % mem_limit
        else:
            return "%iM" % int (mem_limit * 1000)

    def _write_slurm_file(self, job_prefix=None):
        groupname = self.get_global_config().get('groupname', 'kreshuk')
        # read and parse the relevant task config
        task_config = self.get_task_config()
        n_threads = task_config.get("threads_per_job", 1)
        time_limit = self._parse_time_limit(task_config.get("time_limit", 60))
        mem_limit = self._parse_mem_limit(task_config.get("mem_limit", 2))

        # get file paths
        trgt_file = os.path.join(self.tmp_folder, self.task_name + '.py')
        config_tmpl = self._config_path('$1', job_prefix)
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name, job_prefix)
        # TODO set the job-name so that we can parse the squeue output properly
        slurm_template = ("#!/bin/bash\n"
                          "#SBATCH -A %s\n"
                          "#SBATCH -N 1\n"
                          "#SBATCH -n %i\n"
                          "#SBATCH --mem %s\n"
                          "#SBATCH -t %s\n"
                          "%s %s") % (groupname, n_threads,
                                      mem_limit, time_limit,
                                      trgt_file, config_tmpl)
        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % job_name)
        with open(script_path, 'w') as f:
            f.write(slurm_template)

    def prepare_jobs(self, n_jobs, block_list, config, job_prefix=None):
        # write the job configs
        self._write_job_config(n_jobs, block_list, config, job_prefix)
        # write the slurm script file
        self._write_slurm_file(job_prefix)

    def submit_jobs(self, n_jobs, job_prefix=None):
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name, job_prefix)
        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % job_name)
        for job_id in range(n_jobs):
            out_file = os.path.join(self.tmp_folder, 'logs', '%s_%i.log' % (job_name, job_id))
            err_file = os.path.join(self.tmp_folder, 'error_logs', '%s_%i.err' % (job_name, job_id))
            call(['sbatch', '-o', out_file, '-e', err_file, script_path, str(job_id)])

    def wait_for_jobs(self, job_prefix=None):
        # TODO move to some config
        wait_time = 10
        while True:
            time.sleep(wait_time)
            # TODO filter for job name pattern
            n_running = check_output(['squeue -u $USER | wc -l'], shell=True).decode()
            # n_running = check_output(['squeue -u $USER | grep %f | wc -l' % self.task_name], shell=True).decode()
            # need to substract 1 due to header # TODO drop this once we filter for job-name
            n_running = int(n_running.strip('\n')) - 1
            if n_running == 0:
                break


class LocalTask(BaseClusterTask):
    """
    Task for running tasks locally for debugging /
    test purposes
    """

    def prepare_jobs(self, n_jobs, block_list, config, job_prefix=None):
        # write the job configs
        self._write_job_config(n_jobs, block_list, config, job_prefix)

    def _submit(self, job_id, job_prefix):
        script_path = os.path.join(self.tmp_folder, self.task_name + '.py')
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name, job_prefix)
        log_file = os.path.join(self.tmp_folder, 'logs',
                                '%s_%i.log' % (job_name, job_id))
        err_file = os.path.join(self.tmp_folder, 'error_logs',
                                '%s_%i.err' % (job_name, job_id))
        config_file = self._config_path(job_id, job_prefix)
        with open(log_file, 'w') as f_out, open(err_file, 'w') as f_err:
            call([script_path, config_file], stdout=f_out, stderr=f_err)

    def submit_jobs(self, n_jobs, job_prefix=None):
        with futures.ProcessPoolExecutor(n_jobs) as pp:
            tasks = [pp.submit(self._submit, job_id, job_prefix) for job_id in range(n_jobs)]
            [t.result() for t in tasks]

    # don't need to wait for process pool
    def wait_for_jobs(self, job_prefix=None):
        pass


class LSFTask(BaseClusterTask):
    """
    Task for cluster with LSF scheduling system
    (tested on Janelia cluster)
    """

    def prepare_jobs(self, n_jobs, block_list, config, job_prefix=None):
        # write the job configs
        self._write_job_config(n_jobs, block_list, config, job_prefix)

    def submit_jobs(self, n_jobs, job_prefix=None):
        # read the task config to get number of threads and time limit
        task_config = self.get_task_config()
        n_threads = task_config.get("threads_per_job", 1)
        time_limit = task_config.get("time_limit", 60)
        #
        script_path = os.path.join(self.tmp_folder, self.task_name + '.py')
        assert os.path.exists(script_path), script_path
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name, job_prefix)

        for job_id in range(n_jobs):
            config_file = self._config_path(job_id, job_prefix)
            command = '%s %s' % (script_path, config_file)
            log_file = os.path.join(self.tmp_folder, 'logs',
                                    '%s_%i.log' % (job_name, job_id))
            err_file = os.path.join(self.tmp_folder, 'error_logs',
                                    '%s_%i.err' % (job_name, job_id))
            bsub_command = 'bsub -n %i -J %s_%i -We %i -o %s -e %s \'%s\'' % (n_threads, self.task_name,
                                                                              job_id, time_limit,
                                                                              log_file, err_file,
                                                                              command)
            call([bsub_command], shell=True)

    def wait_for_jobs(self, job_prefix=None):
        # TODO move to some config
        wait_time = 10
        while True:
            time.sleep(wait_time)
            # TODO filter for job name pattern
            n_running = check_output(['bjobs | grep $USER | wc -l'], shell=True).decode()
            n_running = int(n_running.strip('\n'))
            if n_running == 0:
                break


class WorkflowBase(luigi.Task):
    """
    Base class for a workflow task, that just chains together
    a workflow of multiple tasks.
    """
    # temporary folder for configurations etc
    tmp_folder = luigi.Parameter()
    # maximum number of concurrent jobs
    max_jobs = luigi.IntParameter()
    # path for the global configuration
    config_dir = luigi.Parameter()
    # TODO max number of threads per job ?
    # target can be local, slurm, lsf (case insensitive)
    target = luigi.Parameter()

    _target_dict = {'lsf': 'LSF', 'slurm': 'Slurm', 'local': 'Local'}

    def _get_task_name(self, task_base_name):
        target_postfix = self._target_dict[self.target.lower()]
        return task_base_name + target_postfix

    def output(self):
        # we just mirror the target of the last task
        return luigi.LocalTarget(self.input().path)
