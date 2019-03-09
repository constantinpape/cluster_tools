import luigi

from ..cluster_tasks import WorkflowBase
from ..relabel import RelabelWorkflow
from .. import write as write_tasks

from .import two_pass_mws as two_pass_tasks
from .import two_pass_assignments as assignmnent_tasks


class TwoPassMwsWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    offsets = luigi.ListParameter()
    halo = luigi.ListParameter(default=None)
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    assignments_key = 'assignments/two_pass_mws'
    relabel_key = 'assignments/two_pass_mws_relabel'

    def requires(self):
        two_pass_task = getattr(two_pass_tasks, self._get_task_name('TwoPassMws'))
        dep = two_pass_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                            config_dir=self.config_dir, dependency=self.dependency,
                            input_path=self.input_path, input_key=self.input_key,
                            output_path=self.output_path, output_key=self.output_key,
                            mask_path=self.mask_path, mask_key=self.mask_key,
                            offsets=self.offsets, halo=self.halo)
        # return dep
        # we need to relabel to keep it efficient, but we also need to
        # relabel the temporary assignments written in pass two
        dep = RelabelWorkflow(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                              config_dir=self.config_dir, dependency=dep, target=self.target,
                              input_path=self.output_path, input_key=self.output_key,
                              assignment_path=self.output_path, assignment_key=self.relabel_key)
        assignmnent_task = getattr(assignmnent_tasks,
                                   self._get_task_name('TwoPassAssignments'))
        dep = assignmnent_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                               config_dir=self.config_dir, dependency=dep,
                               path=self.output_path, key=self.output_key,
                               assignments_path=self.output_path, assignments_key=self.assignments_key,
                               relabel_key=self.relabel_key)
        write_task = getattr(write_tasks,
                             self._get_task_name('Write'))
        dep = write_task(tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         max_jobs=self.max_jobs,
                         input_path=self.output_path, input_key=self.output_key,
                         output_path=self.output_path, output_key=self.output_key,
                         assignment_path=self.output_path, assignment_key=self.assignments_key,
                         identifier='two-pass-mws',
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(TwoPassMwsWorkflow, TwoPassMwsWorkflow).get_config()
        # NOTE config for write tasks is already in relabel workflow
        configs.update({'two_pass_mws': two_pass_tasks.TwoPassMwsLocal.default_task_config(),
                        'two_pass_assignments': assignmnent_tasks.TwoPassAssignmentsLocal.default_task_config(),
                        **RelabelWorkflow.get_config()})
        return configs
