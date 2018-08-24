import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import find_uniques as unique_tasks
from . import find_labeling as labeling_tasks
from .. import write as write_tasks


class RelabelWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()

    def requires(self):
        unique_task = getattr(unique_tasks,
                              self._get_task_name('FindUniques'))
        t1 = unique_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         input_path=self.input_path,
                         input_key=self.input_key,
                         dependency=self.dependency)

        labeling_task = getattr(labeling_tasks,
                                self._get_task_name('FindLabeling'))
        t2 = labeling_task(tmp_folder=self.tmp_folder,
                           max_jobs=self.max_jobs,
                           config_dir=self.config_dir,
                           input_path=self.input_path,
                           input_key=self.input_key,
                           dependency=t1)

        write_task = getattr(write_tasks,
                             self._get_task_name('Write'))
        t3 = write_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        input_path=self.input_path,
                        input_key=self.input_key,
                        output_path=self.input_path,
                        output_key=self.input_key,
                        identifier='relabel',
                        dependency=t2)
        return t3
