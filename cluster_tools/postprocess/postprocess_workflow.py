import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from ..relabel import RelabelWorkflow
from ..relabel import find_uniques as unique_tasks
from ..node_labels import NodeLabelWorkflow

from . import size_filter_blocks as size_filter_tasks
from . import background_size_filter as bg_tasks
from . import filling_size_filter as filling_tasks
from . import filter_blocks as filter_tasks
from . import id_filter as id_tasks


class SizeFilterWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    size_threshold = luigi.IntParameter()
    hmap_path = luigi.Parameter(default='')
    hmap_key = luigi.Parameter(default='')
    relabel = luigi.BoolParameter(default=True)

    def _bg_filter(self, dep):
        filter_task = getattr(bg_tasks,
                              self._get_task_name('BackgroundSizeFilter'))
        dep = filter_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          input_path=self.input_path,
                          input_key=self.input_key,
                          output_path=self.output_path,
                          output_key=self.output_key,
                          dependency=dep)
        return dep

    def _ws_filter(self, dep):
        filter_task = getattr(filling_tasks,
                              self._get_task_name('FillingSizeFilter'))
        dep = filter_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,input_path=self.input_path,
                          input_key=self.input_key,
                          output_path=self.output_path,
                          output_key=self.output_key,
                          hmap_path=self.hmap_path,
                          hmap_key=self.hmap_key,
                          dependency=dep)
        return dep

    def requires(self):
        un_task = getattr(unique_tasks,
                          self._get_task_name('FindUniques'))
        dep = un_task(tmp_folder=self.tmp_folder,
                      max_jobs=self.max_jobs,
                      config_dir=self.config_dir,
                      input_path=self.input_path,
                      input_key=self.input_key,
                      return_counts=True,
                      dependency=self.dependency)
        sf_task = getattr(size_filter_tasks,
                          self._get_task_name('SizeFilterBlocks'))
        dep = sf_task(tmp_folder=self.tmp_folder,
                      max_jobs=self.max_jobs,
                      config_dir=self.config_dir,
                      input_path=self.input_path,
                      input_key=self.input_key,
                      size_threshold=self.size_threshold,
                      dependency=dep)

        if self.hmap_path == '':
            assert self.hmap_key == ''
            dep = self._bg_filter(dep)
        else:
            assert self.hmap_key != ''
            dep = self._ws_filter(dep)

        if self.relabel:
            dep = RelabelWorkflow(tmp_folder=self.tmp_folder,
                                  max_jobs=self.max_jobs,
                                  config_dir=self.config_dir,
                                  target=self.target,
                                  input_path=self.output_path,
                                  input_key=self.output_key,
                                  dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(SizeFilterWorkflow, SizeFilterWorkflow).get_config()
        configs.update({'size_filter_blocks': size_filter_tasks.SizeFilterBlocksLocal.default_task_config(),
                        'background_size_filter': bg_tasks.BackgroundSizeFilterLocal.default_task_config(),
                        'filling_size_filter': filling_tasks.FillingSizeFilterLocal.default_task_config(),
                        **RelabelWorkflow.get_config()})
        return configs


class FilterLabelsWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    label_path = luigi.Parameter()
    label_key = luigi.Parameter()
    node_label_path = luigi.Parameter()
    node_label_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    filter_labels = luigi.ListParameter()

    def requires(self):
        dep = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                target=self.target, max_jobs=self.max_jobs,
                                ws_path=self.input_path, ws_key=self.input_key,
                                input_path=self.label_path, input_key=self.label_key,
                                output_path=self.node_label_path,
                                output_key=self.node_label_key,
                                prefix='filter_labels', max_overlap=True,
                                dependency=self.dependency)
        id_task = getattr(id_tasks,
                          self._get_task_name('IdFilter'))
        id_filter_path = os.path.join(self.output_path, 'filtered_ids.json')
        dep = id_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                      dependency=dep, max_jobs=self.max_jobs,
                      node_label_path=self.node_label_path,
                      node_label_key=self.node_label_key,
                      output_path = id_filter_path,
                      filter_labels=self.filter_labels)
        filter_task = getattr(filter_tasks,
                              self._get_task_name('FilterBlocks'))
        dep = filter_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                          dependency=dep, max_jobs=self.max_jobs,
                          input_path=self.input_path, input_key=self.input_key,
                          filter_path=id_filter_path,
                          output_path=self.output_path, output_key=self.output_key)
        return dep

    @staticmethod
    def get_config():
        configs = super(FilterLabelsWorkflow, FilterLabelsWorkflow).get_config()
        configs.update({'id_filter':
                        id_tasks.IdFilterLocal.default_task_config(),
                        'filter_blocks':
                        filter_tasks.FilterBlocksLocal.default_task_config(),
                        **NodeLabelWorkflow.get_config()})
        return configs
