import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from ..node_labels import NodeLabelWorkflow
from . import filter_blocks as filter_tasks
from . import id_filter as id_tasks


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
