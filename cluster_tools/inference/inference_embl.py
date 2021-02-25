import os
from cluster_tools.cluster_tasks import SlurmTask
from .inference import InferenceBase


# custom slurm submission for gpu set-up at embl
# see https://github.com/kreshuklab/gpu_envs for details
class InferenceEmbl(InferenceBase, SlurmTask):
    def _write_slurm_file(self, job_prefix=None):
        global_config = self.get_global_config()
        groupname = global_config.get('groupname', None)

        # read and parse the relevant task config
        task_config = self.get_task_config()
        n_threads = task_config.get("threads_per_job", 1)
        time_limit = self._parse_time_limit(task_config.get("time_limit", 60))
        mem_limit = self._parse_mem_limit(task_config.get("mem_limit", 2))
        gpu_type = task_config['gpu_type']

        # and a local qos that overrides the global one
        qos = task_config.get("qos", "normal")

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
        slurm_template += "#SBATCH -p gpu\n"
        slurm_template += "#SBATCH --gres=gpu:1\n"
        slurm_template += "#SBATCH -C gpu=%s\n" % gpu_type
        # add the groupname if specified
        if groupname is not None:
            slurm_template += "#SBATCH -A %s\n" % groupname

        # slurm directives are done
        slurm_template += "\n"
        slurm_template += "module purge\n"
        slurm_template += "module load PyTorch\n"
        slurm_template += "source activate_gpu_env\n"
        slurm_template += "$GPU_PY %s %s" % (trgt_file, config_tmpl)

        script_path = os.path.join(self.tmp_folder, 'slurm_%s.sh' % job_name)
        with open(script_path, 'w') as f:
            f.write(slurm_template)
