import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import watershed as watershed_tasks
from ..relabel import RelabelWorkflow


class WatershedWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    def requires(self):
        ws_task = getattr(watershed_tasks,
                          self._get_task_name('Watershed'))
        t1 = ws_task(tmp_folder=self.tmp_folder,
                     max_jobs=self.max_jobs,
                     config_dir=self.config_dir,
                     input_path=self.input_path,
                     input_key=self.input_key,
                     output_path=self.output_path,
                     output_key=self.output_key)
        t2 = RelabelWorkflow(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             target=self.target,
                             input_path=self.output_path,
                             input_key=self.output_key,
                             dependency=t1)
        return t2

    @staticmethod
    def get_config():
        configs = super(WatershedWorkflow, WatershedWorkflow).get_config()
        configs.update({'watershed': watershed_tasks.WatershedLocal.default_task_config()})
        return configs
