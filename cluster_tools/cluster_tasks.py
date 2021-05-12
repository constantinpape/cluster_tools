import os
import shutil
import stat
import json
import time
import fileinput
import sys
from copy import deepcopy
from concurrent import futures
from subprocess import call, check_output, CalledProcessError, STDOUT
from datetime import datetime, timedelta
from multiprocessing import cpu_count

import numpy as np
import luigi

from .utils.parse_utils import parse_blocks_task, parse_job, parse_job_lsf
from .utils.task_utils import DummyTask


class FailedJobsError(Exception):
    """ Custom exception for failed jobs
    """
    pass


class BaseClusterTask(luigi.Task):
    """
    Base class for a task to run on the cluster.

    Subclasses need to implement
    a run function: `run_impl`.
    The `run_impl` method executes the job and must use the pre-implemented API
    functions in the correct order, see example below.

    Example:
        def run_impl(self):
            # initialize the tmp folder and log dirs
            shebang, block_shape, roi_begin, roi_end = self.global_config_values()
            self.init(shebang) # must-call

            # read config file and get number of n_jobs
            config = self.get_task_config()

            # need to get spatial shape corresponding to this block
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
            n_jobs = min(self.max_jobs, len(block_list))

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
    # variables for retry mechanism
    # allow retries for this task, set to false in deriving class if retries not allowed
    allow_retry = True
    # number of retries already done
    n_retries = 0

    #
    # API
    #

    def run(self):
        self.make_dirs()
        self._write_log("Start task %s" % self.task_name)
        try:
            self.run_impl()
        # if a failed jobs error was raised, one or more jobs failed
        # and the log file was moved already
        except FailedJobsError as e:
            raise e
        # otherwise the exception happened before submitting jobs and
        # we need to move the log file
        except Exception as e:
            out_path = self.output().path
            fail_path = out_path[:-4] + '_failed.log'
            msg = str(e.message) if hasattr(e, 'message') else str(e)
            self._write_log("task failed in `run_impl` with %s" % msg)
            self._write_log("move log from %s to %s" % (out_path, fail_path))
            shutil.move(out_path, fail_path)
            raise e
        self._write_log("Done task %s" % self.task_name)

    def init(self, shebang):
        """ Init tmp dir and python scripts.

        Should be the first call in `run_impl`.
        """
        self._write_script_file(shebang)

    @staticmethod
    def parse_jobs(log_prefix, max_jobs):
        passed_jobs = []
        for job_id in range(max_jobs):
            path = log_prefix + '%i.log' % job_id
            if parse_job(path, job_id):
                passed_jobs.append(job_id)
        return passed_jobs

    def check_jobs(self, n_jobs, job_prefix=None):
        """ Check for jobs that ran successfully
        """
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name,
                                                                        job_prefix)
        log_prefix = os.path.join(self.tmp_folder, 'logs', '%s_' % job_name)
        success_list = self.parse_jobs(log_prefix, n_jobs)

        if len(success_list) == n_jobs:
            self._write_log("%s finished successfully" % self.task_name)
        else:
            failed_jobs = set(range(n_jobs)) - set(success_list)
            self._write_log("%s failed for jobs:" % self.task_name)
            self._write_log("%s" % ', '.join(map(str, failed_jobs)))

            # For slurm, we also write out the failed slurm ids
            if hasattr(self, 'slurm_ids'):
                self._write_log("corresponds to failed slurm ids:")
                failed_slurm_ids = [self.slurm_ids[fjob_id] for fjob_id in list(failed_jobs)]
                self._write_log("%s" % ', '.join(map(str, failed_slurm_ids)))

            # check if conditions to retry jobs are met
            max_num_retries = self.get_global_config().get('max_num_retries', 0)
            # does the number of retries exceed the max number of retries?
            # does this task allow for retries?
            retry = (self.n_retries < max_num_retries) and self.allow_retry
            # have at least 50 % of the jobs passed?
            # we use this as heuristic to determine if something is fundementally broken.
            retry = retry and len(failed_jobs) / n_jobs < 0.5

            if retry:
                failed_blocks = self.get_failed_blocks(n_jobs, success_list, job_prefix)
                self._write_log("resubmitting %i failed blocks in %i retry attempt" % (len(failed_blocks),
                                                                                       self.n_retries + 1))
                self.n_retries += 1
                self.block_list = failed_blocks
                self.run()
            else:
                # rename log file due to fail
                out_path = self.output().path
                fail_path = out_path[:-4] + '_failed.log'
                self._write_log("move log from %s to %s" % (out_path, fail_path))
                shutil.move(out_path, fail_path)
                raise FailedJobsError("Task: %s failed for %i / %i jobs" % (self.task_name,
                                                                            len(failed_jobs),
                                                                            n_jobs))

    def get_failed_blocks(self, n_jobs, passed_jobs=[], job_prefix=None):
        """ Parse the log of failed jobs to find the ids of all blocks that have failed.
        """
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name,
                                                                        job_prefix)
        # for the jobs that have completely passed, we can add the block list from the config
        passed_blocks = []
        for job_id in passed_blocks:
            config_path = self._config_path(job_id, job_prefix)
            with open(config_path, 'r') as f:
                passed_blocks.extend(json.load(f)['block_list'])

        # for the failed jobs, we parse the output logs
        log_prefix = os.path.join(self.tmp_folder, 'logs', '%s_' % job_name)
        passed_blocks.extend(parse_blocks_task(log_prefix, n_jobs, passed_jobs))

        # return the list of failed blocks
        return list(set(self.block_list) - set(passed_blocks))

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
        return {"threads_per_job": 1, "time_limit": 60, "mem_limit": 1., "qos": "normal", "slurm_requirements": []}

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
                "shebang": sys.executable,
                "roi_begin": None,
                "roi_end": None,
                "groupname": None,
                "partition": None,
                "max_num_retries": 0,
                "block_list_path": None,
                "easybuild": True,
                "qos": "normal"}

    def global_config_values(self, with_block_list_path=False):
        """ Load the global config values that are needed
            in most of the tasks
        """
        config = self.get_global_config()
        # first two return values (shebang, block_shape) must exist
        # the rest default to None
        conf = (config["shebang"], config["block_shape"],
                config.get("roi_begin", None),
                config.get("roi_end", None))
        if with_block_list_path:
            conf = conf + (config.get("block_list_path", None),)
        return conf

    def clean_up_for_retry(self, block_list, prefix=None):
        """ Clean up before starting a retry.
        The base implementation is just a dummy.
        """
        pass

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, self.task_name + '.log'))

    #
    # Must implement API
    #

    def prepare_jobs(self, n_jobs, block_list, config, job_prefix=None, consecutive_blocks=False):
        raise NotImplementedError("BaseClusterTask does not implement this functionality")

    def submit_jobs(self, n_jobs, job_prefix=None):
        raise NotImplementedError("BaseClusterTask does not implement this functionality")

    def wait_for_jobs(self, job_prefix=None):
        raise NotImplementedError("BaseClusterTask does not implement this functionality")

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
        os.makedirs(self.tmp_folder, exist_ok=True)
        os.makedirs(os.path.join(self.tmp_folder, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_folder, 'error_logs'), exist_ok=True)
        self._write_log('created tmp-folder and log dirs @ %s' % self.tmp_folder)

    def _write_single_job_config(self, config, job_prefix):
        config_path = self._config_path(0, job_prefix)
        with open(config_path, 'w') as f:
            json.dump(config, f)

    def _write_multiple_job_configs(self, n_jobs, block_list, config, job_prefix,
                                    consecutive_blocks):

        # TODO there must be a more elegant way of doing this
        if consecutive_blocks:
            # distribute blocks to jobs as equal as possible
            blocks_per_job = np.zeros(n_jobs, dtype='uint32')
            block_count = len(block_list)
            job_id = 0
            while block_count > 0:
                blocks_per_job[job_id] += 1
                block_count -= 1
                job_id += 1
                job_id = job_id % n_jobs

            assert np.sum(blocks_per_job) == len(block_list)
            # make prepartition
            prepartiion = []
            block_id = 0
            for bpj in blocks_per_job:
                prepartiion.append(list(range(block_id, block_id + bpj)))
                block_id += bpj

        # write the configurations for all jobs to the tmp folder
        for job_id in range(n_jobs):
            # if `consecutive_blocks` is true, we keep the block_ids in
            # block_jobs consecutive
            if consecutive_blocks:
                block_jobs = prepartiion[job_id]
            else:
                block_jobs = block_list[job_id::n_jobs]
            job_config = {'block_list': block_jobs, **config}
            config_path = self._config_path(job_id, job_prefix)
            with open(config_path, 'w') as f:
                json.dump(job_config, f)

    def _write_job_config(self, n_jobs, block_list, config,
                          job_prefix=None, consecutive_blocks=False):
        # check f we have a reduce style block, that is
        # not distributed over blocks
        if block_list is None:
            assert n_jobs == 1
            self._write_single_job_config(config, job_prefix)
        # otherwise, we have multiple jobs distributed over blocks
        else:
            # we add the block list to this class to know all the blocks
            # that were scheduled if we need to rerun this task
            self.block_list = block_list
            self._write_multiple_job_configs(n_jobs, block_list, config,
                                             job_prefix, consecutive_blocks)
        self._write_log('written config for %i jobs' % n_jobs)

    # copy the python script to the temp folder and replace the shebang
    def _write_script_file(self, shebang):
        assert os.path.exists(self.src_file), self.src_file
        trgt_file = os.path.join(self.tmp_folder, self.task_name + '.py')
        shutil.copy(self.src_file, trgt_file)

        # check that the shebang/executable is valid
        if shebang.startswith('#!'):
            executable = shebang.lstrip('#!').lstrip()
        else:
            executable = deepcopy(shebang)
            shebang = "#! " + shebang
        if not os.path.exists(executable):
            raise RuntimeError("The python executable %s is not valid" % executable)

        # not necessary on windows
        if os.name != 'nt':
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
            return "%iG" % mem_limit
        else:
            return "%iM" % int(mem_limit * 1000)

    def _write_slurm_file(self, job_prefix=None):
        global_config = self.get_global_config()
        groupname = global_config.get('groupname', None)
        partition = global_config.get('partition', None)
        easybuild = global_config.get("easybuild", True)

        # read and parse the relevant task config
        task_config = self.get_task_config()
        n_threads = task_config.get("threads_per_job", 1)
        time_limit = self._parse_time_limit(task_config.get("time_limit", 60))
        mem_limit = self._parse_mem_limit(task_config.get("mem_limit", 2))

        # TODO we should be able to set a global qos
        # and a local qos that overrides the global one
        qos = global_config.get("qos", "normal")

        # additional job requirements
        requirements = task_config.get("slurm_requirements", [])

        # get file paths
        trgt_file = os.path.join(self.tmp_folder, self.task_name + '.py')
        config_tmpl = self._config_path('$1', job_prefix)
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name,
                                                                        job_prefix)
        slurm_template = ("#!/bin/bash\n"
                          "#SBATCH -N 1\n"
                          "#SBATCH -c %i\n"
                          "#SBATCH --mem %s\n"
                          "#SBATCH -t %s\n"
                          "#SBATCH --qos=%s\n") % (n_threads, mem_limit, time_limit, qos)
        # add the groupname if specified
        if groupname is not None:
            slurm_template += "#SBATCH -A %s\n" % groupname
        # add the partition (= which queue should be used, if specified)
        if partition is not None:
            slurm_template += "#SBATCH -p %s\n" % partition
        for req in requirements:
            slurm_template += "#SBATCH -C %s\n" % req

        # slurm directives are done
        slurm_template += "\n"
        # do we have easybuild ?
        if easybuild:
            slurm_template += "module purge\n"
            slurm_template += "module load GCC\n"
        slurm_template += ("%s %s") % (trgt_file, config_tmpl)

        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % job_name)
        with open(script_path, 'w') as f:
            f.write(slurm_template)

    def prepare_jobs(self, n_jobs, block_list, config,
                     job_prefix=None, consecutive_blocks=False):
        # write the job configs
        self._write_job_config(n_jobs, block_list, config, job_prefix, consecutive_blocks)
        # write the slurm script file
        self._write_slurm_file(job_prefix)

    def submit_jobs(self, n_jobs, job_prefix=None):
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name,
                                                                        job_prefix)
        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % job_name)
        self.slurm_ids = []
        for job_id in range(n_jobs):
            out_file = os.path.join(self.tmp_folder, 'logs', '%s_%i.log' % (job_name, job_id))
            err_file = os.path.join(self.tmp_folder, 'error_logs', '%s_%i.err' % (job_name,
                                                                                  job_id))
            command = ['sbatch', '-o', out_file, '-e', err_file, '-J',
                       '%s_%i' % (job_name, job_id), script_path, str(job_id)]
            # call(command)
            outp = check_output(command).decode().rstrip()
            # get the slurm job-id
            # NOTE: slurm ids are not always integer, so we cannot cast to int here
            slurm_id = outp.split()[-1]
            self.slurm_ids.append(slurm_id)
            # print slurm message
            print(outp)

    def wait_for_jobs(self, job_prefix=None):
        # TODO move to some config
        wait_time = 10
        while True:
            time.sleep(wait_time)

            try:
                outp = check_output(['squeue -u $USER | grep $USER'], shell=True).decode()
            except CalledProcessError as e:
                # handle error for empty queue
                outp = e.output.decode().rstrip()
                if outp == '':
                    break
                else:
                    raise e

            outp = [out for out in outp.split('\n') if out != '']
            # check how many jobs there are in total, if none, stop waiting
            n_running = len(outp)
            if n_running == 0:
                break
            # if we have jobs, check how many belong to this task
            n_running = sum([out.split()[0] in self.slurm_ids for out in outp])
            if n_running == 0:
                break


class LocalTask(BaseClusterTask):
    """
    Task for running tasks locally via sub-processes
    """
    # don't want to start too many local jobs, because
    # this is usually a sign that forgot to set the target
    # to slurm or lsf
    max_local_jobs = cpu_count()

    def prepare_jobs(self, n_jobs, block_list, config,
                     job_prefix=None, consecutive_blocks=False):
        # write the job configs
        self._write_job_config(n_jobs, block_list, config, job_prefix, consecutive_blocks)

    # the normal submission logic doesn't work on windows
    def _submit_win(self, script_path, config_file, log_file, err_file):
        with open(log_file, 'w') as f_out, open(err_file, 'w') as f_err:
            assert os.path.exists(script_path), script_path
            call(["python", script_path, config_file], stdout=f_out, stderr=f_err, shell=True)

    def _submit_unix(self, script_path, config_file, log_file, err_file):
        with open(log_file, 'w') as f_out, open(err_file, 'w') as f_err:
            assert os.path.exists(script_path), script_path
            call([script_path, config_file], stdout=f_out, stderr=f_err)

    def _submit(self, job_id, job_prefix):
        script_path = os.path.join(self.tmp_folder, self.task_name + '.py')
        assert os.path.exists(script_path), script_path
        config_file = self._config_path(job_id, job_prefix)
        assert os.path.exists(config_file), config_file

        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name,
                                                                        job_prefix)
        log_file = os.path.join(self.tmp_folder, 'logs',
                                '%s_%i.log' % (job_name, job_id))
        err_file = os.path.join(self.tmp_folder, 'error_logs',
                                '%s_%i.err' % (job_name, job_id))
        if os.name == 'nt':
            self._submit_win(script_path, config_file, log_file, err_file)
        else:
            self._submit_unix(script_path, config_file, log_file, err_file)

    def submit_jobs(self, n_jobs, job_prefix=None):
        assert n_jobs <= self.max_local_jobs,\
            "Trying to submit %i local jobs but limit is %i. Did you forget to set the target to slurm or lsf?" %\
            (n_jobs, self.max_local_jobs)
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
    def prepare_jobs(self, n_jobs, block_list, config,
                     job_prefix=None, consecutive_blocks=False):
        # write the job configs
        self._write_job_config(n_jobs, block_list, config, job_prefix, consecutive_blocks)

    def submit_jobs(self, n_jobs, job_prefix=None):
        # read the task config to get number of threads and time limit
        task_config = self.get_task_config()
        n_threads = task_config.get("threads_per_job", 1)
        time_limit = task_config.get("time_limit", 60)
        #
        script_path = os.path.join(self.tmp_folder, self.task_name + '.py')
        assert os.path.exists(script_path), script_path

        self.bsub_ids = []
        job_name = self.task_name if job_prefix is None else '%s_%s' % (self.task_name,
                                                                        job_prefix)

        for job_id in range(n_jobs):
            config_file = self._config_path(job_id, job_prefix)
            command = '%s %s' % (script_path, config_file)
            log_file = os.path.join(self.tmp_folder, 'logs',
                                    '%s_%i.log' % (job_name, job_id))
            err_file = os.path.join(self.tmp_folder, 'error_logs',
                                    '%s_%i.err' % (job_name, job_id))
            bsub_command = 'bsub -n %i -J %s_%i -We %i -o %s -e %s \'%s\'' % (n_threads,
                                                                              self.task_name,
                                                                              job_id, time_limit,
                                                                              log_file, err_file,
                                                                              command)
            # call([bsub_command], shell=True)
            # submit job and get the bsub job id from its output
            outp = check_output([bsub_command], shell=True).decode().rstrip()
            bsub_id = int(outp.split()[1].lstrip('<').rstrip('>'))
            self.bsub_ids.append(bsub_id)
            print(outp)

    def wait_for_jobs(self, job_prefix=None):
        # TODO move to some config
        wait_time = 10
        while True:
            time.sleep(wait_time)
            # parse the output from bjobs
            try:
                outp = check_output(['bjobs | grep $USER'], shell=True, stderr=STDOUT).decode()
            # check if no jobs are in queue (throws an error we need to capture)
            except CalledProcessError as e:
                outp = e.output.decode().rstrip()
                if outp == 'No unfinished job found':
                    break
                else:
                    raise e

            outp = outp.split('\n')
            outp = [out for out in outp if out != '']
            # check how many jobs there are in total, if none, stop waiting
            n_running = len(outp)
            if n_running == 0:
                break
            # if we have jobs, check how many belong to this task
            n_running = sum([int(out.split()[0]) in self.bsub_ids for out in outp])
            if n_running == 0:
                break

    # need to override this for lsf
    @staticmethod
    def parse_jobs(log_prefix, max_jobs):
        passed_jobs = []
        for job_id in range(max_jobs):
            path = log_prefix + '%i.log' % job_id
            if parse_job_lsf(path, job_id):
                passed_jobs.append(job_id)
        return passed_jobs

    # TODO I think LSF appends to the output and logfile
    # so we need to clean them up here in order to have clean logs
    def clean_up_for_retry(self, block_list, prefix=None):
        """ Clean up before starting a retry.
        """
        pass


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
    # target can be local, slurm, lsf (case insensitive)
    target = luigi.Parameter()
    # the workflow can have dependencies; per default we
    # set to be a dummy task that is always successfull
    dependency = luigi.TaskParameter(default=DummyTask())

    _target_dict = {'lsf': 'LSF', 'slurm': 'Slurm', 'local': 'Local'}

    def _get_task_name(self, task_base_name):
        target_postfix = self._target_dict[self.target.lower()]
        return task_base_name + target_postfix

    def output(self):
        # we just mirror the target of the last task
        return luigi.LocalTarget(self.input().path)

    @staticmethod
    def get_config():
        """ Return all default configs and their save_path indexed by the task name
        """
        return {'global': BaseClusterTask.default_global_config()}
