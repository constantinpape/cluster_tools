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
    ws_config_path = luigi.Parameter()

    def requires(self):
        ws_task = getattr(watershed_tasks,
                          self._get_task_name('Watershed'))
        t1 = ws_task(tmp_folder=self.tmp_folder,
                     max_jobs=self.max_jobs,
                     global_config_path=self.global_config_path,
                     input_path=self.input_path,
                     input_key=self.input_key,
                     output_path=self.output_path,
                     output_key=self.output_key,
                     ws_config_path=self.ws_config_path)
        t2 = RelabelWorkflow(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             global_config_path=self.global_config_path,
                             target=self.target,
                             input_path=self.output_path,
                             input_key=self.output_key,
                             dependency=t1)
        return t2
