import os
import luigi
from .find_uniques import FindUniquesTask
from .find_labeling import FindLabelingTask
from ..write import WriteAssignmentTask


class RelabelWorkflow(luigi.Task):

    # path to the n5 file and keys
    path = luigi.Parameter()
    key = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    # path to the configuration
    # TODO allow individual paths for individual blocks
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    # TODO different time estimates for different sub-tasks
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        uniques_task = FindUniquesTask(path=self.path,
                                       key=self.key,
                                       max_jobs=self.max_jobs,
                                       config_path=self.config_path,
                                       tmp_folder=self.tmp_folder,
                                       dependency=self.dependency,
                                       time_estimate=self.time_estimate,
                                       run_local=self.run_local)
        labels_task = FindLabelingTask(path=self.path,
                                       key=self.key,
                                       config_path=self.config_path,
                                       tmp_folder=self.tmp_folder,
                                       dependency=uniques_task,
                                       time_estimate=self.time_estimate,
                                       run_local=self.run_local)
        write_task = WriteAssignmentTask(path=self.path,
                                         in_key=self.key,
                                         out_key=self.key,
                                         config_path=self.config_path,
                                         max_jobs=self.max_jobs,
                                         tmp_folder=self.tmp_folder,
                                         identifier='write_relabel',
                                         dependency=labels_task,
                                         time_estimate=self.time_estimate,
                                         run_local=self.run_local)
        return write_task

    def run(self):
        out_path = self.input().path
        assert os.path.exists(out_path)
        res_path = self.output().path
        with open(res_path, 'w') as f:
            f.write("Success")

    def output(self):
        out_file = os.path.join(self.tmp_folder, 'relabeling_workflow.log')
        return luigi.LocalTarget(out_file)
