import luigi

from cluster_tools.cluster_tasks import WorkflowBase
from cluster_tools import write as write_tasks
from . import resolve_individual_objects as resolve_tasks


class ResolvingWorkflow(WorkflowBase):
    problem_path = luigi.Parameter()
    path = luigi.Parameter()
    output_path = luigi.Parameter()
    objects_group = luigi.Parameter()
    assignment_in_key = luigi.Parameter()
    assignment_out_key = luigi.Parameter()

    ws_key = luigi.Parameter(default='')
    out_key = luigi.Parameter(default='')

    def requires(self):
        resolve_task = getattr(resolve_tasks, self._get_task_name('ResolveIndividualObjects'))
        dep = resolve_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir, max_jobs=self.max_jobs,
                           problem_path=self.problem_path, objects_path=self.problem_path,
                           objects_group=self.objects_group,
                           assignment_in_path=self.path, assignment_in_key=self.assignment_in_key,
                           assignment_out_path=self.output_path, assignment_out_key=self.assignment_out_key,
                           dependency=self.dependency)

        if self.out_key != '':
            assert self.ws_key != ''
            write_task = getattr(write_tasks, self._get_task_name('Write'))
            dep = write_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir, max_jobs=self.max_jobs,
                             input_path=self.path, input_key=self.ws_key,
                             output_path=self.output_path, output_key=self.out_key,
                             assignment_path=self.output_path, assignment_key=self.assignment_out_key,
                             identifier='resolve_inidividual_objects', dependency=dep)

        return dep

    @staticmethod
    def get_config():
        config = super(ResolvingWorkflow, ResolvingWorkflow).get_config()
        config.update({'write': write_tasks.WriteLocal.default_task_config(),
                       'resolve_inidividual_objects':
                       resolve_tasks.ResolveIndividualObjectsLocal.default_task_config()})
        return config
