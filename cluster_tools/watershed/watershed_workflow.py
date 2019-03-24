import luigi

from ..cluster_tasks import WorkflowBase
from . import watershed as watershed_tasks
from . import two_pass_watershed as two_pass_tasks
from . import agglomerate as agglomerate_tasks
from ..relabel import RelabelWorkflow


class WatershedWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')
    two_pass = luigi.BoolParameter(default=False)
    agglomeration = luigi.BoolParameter(default=False)

    def requires(self):
        if self.two_pass:
            ws_task = getattr(two_pass_tasks,
                              self._get_task_name('TwoPassWatershed'))
        else:
            ws_task = getattr(watershed_tasks,
                              self._get_task_name('Watershed'))
        dep = ws_task(tmp_folder=self.tmp_folder,
                      max_jobs=self.max_jobs,
                      config_dir=self.config_dir,
                      input_path=self.input_path,
                      input_key=self.input_key,
                      output_path=self.output_path,
                      output_key=self.output_key,
                      mask_path=self.mask_path,
                      mask_key=self.mask_key)

        # run post-ws agglomeration if specified
        if self.agglomeration:
            agglomerate_task = getattr(agglomerate_tasks,
                                       self._get_task_name('Agglomerate'))
            dep = agglomerate_task(tmp_folder=self.tmp_folder,
                                   max_jobs=self.max_jobs,
                                   config_dir=self.config_dir,
                                   dependency=dep,
                                   input_path=self.input_path,
                                   input_key=self.input_key,
                                   output_path=self.output_path,
                                   output_key=self.output_key,
                                   have_ignore_label=self.mask_path != '')

        dep = RelabelWorkflow(tmp_folder=self.tmp_folder,
                              max_jobs=self.max_jobs,
                              config_dir=self.config_dir,
                              target=self.target,
                              input_path=self.output_path,
                              input_key=self.output_key,
                              assignment_path=self.output_path,
                              assignment_key='relabel_watershed',
                              dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(WatershedWorkflow, WatershedWorkflow).get_config()
        configs.update({'watershed': watershed_tasks.WatershedLocal.default_task_config(),
                        'two_pass_watershed': two_pass_tasks.TwoPassWatershedLocal.default_task_config(),
                        'agglomerate': agglomerate_tasks.AgglomerateLocal.default_task_config(),
                        **RelabelWorkflow.get_config()})
        return configs
