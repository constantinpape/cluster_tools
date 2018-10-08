import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from ..relabel import RelabelWorkflow
from ..relabel import find_uniques as unique_tasks
from . import size_filter_blocks as size_filter_tasks
from . import background_size_filter as bg_tasks
from . import filling_size_filter as filling_tasks


class SizeFilterWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    size_threshold = luigi.IntParameter()
    hmap_path = luigi.Parameter(default='')
    hmap_key = luigi.Parameter(default='')
    relabel = luigi.BoolParameter(default=True)

    def requires(self):
        un_task = getattr(unique_tasks,
                          self._get_task_name('FindUniques'))
        t1 = un_task(tmp_folder=self.tmp_folder,
                     max_jobs=self.max_jobs,
                     config_dir=self.config_dir,
                     input_path=self.input_path,
                     input_key=self.input_key,
                     return_counts=True,
                     dependency=self.dependency)
        sf_task = getattr(size_filter_tasks,
                          self._get_task_name('SizeFilterBlocks'))
        t2 = sf_task(tmp_folder=self.tmp_folder,
                     max_jobs=self.max_jobs,
                     config_dir=self.config_dir,
                     input_path=self.input_path,
                     input_key=self.input_key,
                     size_threshold=self.size_threshold,
                     dependency=t1)

        if self.hmap_path == '':
            assert self.hmap_key == ''
            filter_task = getattr(bg_tasks,
                                  self._get_task_name('BackgroundSizeFilter'))
            t3 = filter_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             input_path=self.input_path,
                             input_key=self.input_key,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             dependency=t2)

        else:
            filter_task = getattr(filling_tasks,
                                  self._get_task_name('FillingSizeFilter'))
            t3 = filter_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,input_path=self.input_path,
                             input_key=self.input_key,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             hmap_path=self.hmap_path,
                             hmap_key=self.hmap_key,
                             dependency=t2)

        if self.relabel:
            t4 = RelabelWorkflow(tmp_folder=self.tmp_folder,
                                 max_jobs=self.max_jobs,
                                 config_dir=self.config_dir,
                                 target=self.target,
                                 input_path=self.output_path,
                                 input_key=self.output_key,
                                 dependency=t3)
            return t4
        else:
            return t3

    @staticmethod
    def get_config():
        configs = super(SizeFilterWorkflow, SizeFilterWorkflow).get_config()
        # TODO properly set configs
        # configs.update({'watershed': watershed_tasks.WatershedLocal.default_task_config()})
        return configs
