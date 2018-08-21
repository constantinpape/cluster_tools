import os
import json
import luigi

import .find_uniques as unique_tasks
import .find_labeling as labeling_tasks
import ..write import as write_tasks


class RelabelWorkflow(luigi.Task):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        unique_task = getattr(unique_tasks,
                              self._get_task_name('FindUniques'))
        t1 = unique_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         global_config_path=self.global_config_path,
                         input_path=self.input_path,
                         input_key=self.input_key,
                         dependency=self.dependency)

        labeling_task = getattr(labeling_tasks,
                                self._get_task_name('FindLabeling'))
        t2 = labeling_task(tmp_folder=self.tmp_folder,
                           max_jobs=self.max_jobs,
                           global_config_path=self.global_config_path,
                           input_path=self.input_path,
                           input_key=self.input_key,
                           dependency=t1)

        write_task = getattr(write_tasks,
                             self._get_task_name('Write'))
        t3 = write_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        global_config_path=self.global_config_path,
                        input_path=self.input_path,
                        input_key=self.input_key,
                        output_path=self.input_path,
                        output_key=self.input_key,
                        identifier='relabel',
                        dependency=t2)
        return t3
