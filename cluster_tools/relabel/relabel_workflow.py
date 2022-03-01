import luigi

from ..cluster_tasks import WorkflowBase
from . import find_uniques as unique_tasks
from . import find_labeling as labeling_tasks
from . import merge_uniques as merge_tasks
from .. import write as write_tasks


class RelabelWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    output_path = luigi.Parameter(default="")
    output_key = luigi.Parameter(default="")
    prefix = luigi.Parameter(default=None)

    def requires(self):
        unique_task = getattr(unique_tasks,
                              self._get_task_name("FindUniques"))
        dep = unique_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          input_path=self.input_path,
                          input_key=self.input_key,
                          dependency=self.dependency,
                          prefix=self.prefix)

        # for now, we hard-code the assignment path here,
        # because it is only used internally for this task
        # but it could also be exposed if this is useful
        # at some point
        labeling_task = getattr(labeling_tasks, self._get_task_name("FindLabeling"))
        dep = labeling_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                            config_dir=self.config_dir, dependency=dep,
                            input_path=self.input_path, input_key=self.input_key,
                            assignment_path=self.assignment_path,
                            assignment_key=self.assignment_key,
                            prefix=self.prefix)

        # check if we relabel in-place (default) or to a new output file
        if self.output_path == "":
            out_path = self.input_path
            out_key = self.input_key
        else:
            assert self.output_key != ""
            out_path = self.output_path
            out_key = self.output_key

        write_task = getattr(write_tasks, self._get_task_name("Write"))
        write_id = "relabel" if self.prefix is None else f"relabel_{self.prefix}"
        dep = write_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         input_path=self.input_path,
                         input_key=self.input_key,
                         output_path=out_path,
                         output_key=out_key,
                         assignment_path=self.assignment_path,
                         assignment_key=self.assignment_key,
                         identifier=write_id,
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(RelabelWorkflow, RelabelWorkflow).get_config()
        configs.update({"find_uniques":
                        unique_tasks.FindUniquesLocal.default_task_config(),
                        "find_labeling":
                        labeling_tasks.FindLabelingLocal.default_task_config(),
                        "write":
                        write_tasks.WriteLocal.default_task_config()})
        return configs


class UniqueWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    def requires(self):
        unique_task = getattr(unique_tasks,
                              self._get_task_name("FindUniques"))
        dep = unique_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          input_path=self.input_path,
                          input_key=self.input_key,
                          dependency=self.dependency)

        merge_task = getattr(merge_tasks,
                             self._get_task_name("MergeUniques"))
        dep = merge_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                         config_dir=self.config_dir, dependency=dep,
                         input_path=self.input_path, input_key=self.input_key,
                         output_path=self.output_path, output_key=self.output_key)
        return dep

    @staticmethod
    def get_config():
        configs = super(UniqueWorkflow, UniqueWorkflow).get_config()
        configs.update({"find_uniques":
                        unique_tasks.FindUniquesLocal.default_task_config(),
                        "merge_uniques":
                        merge_tasks.MergeUniquesLocal.default_task_config()})
        return configs
