import os
import json
import luigi

from .. cluster_tasks import WorkflowBase
from . import block_node_labels as label_tasks
# TODO implement merge task
# from . import merge_node_overlaps as merge_tasks


# TODO implement proper scalable workflow
class NodeOverlapWorkflow(WorkflowBase):
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    def requires(self):
        label_task = getattr(label_tasks,
                             self._get_task_name('BlockNodeLabels'))
        t1 = label_task(max_jobs=self.max_jobs,
                        tmp_folder=self.tmp_folder,
                        config_dir=self.config_dir,
                        dependency=self.dependency,
                        labels_path=self.labels_path,
                        labels_key=self.labels_key,
                        input_path=self.input_path,
                        input_key=self.input_key,
                        output_path=self.output_path,
                        output_key=self.output_key)
        return t1
        # TODO implement merge task
        # merge_task = getattr(merge_tasks,
        #                      self._get_task_name('MergeEdgeLabels'))
        # t2 = merge_task(max_jobs=self.max_jobs,
        #                 tmp_folder=self.tmp_folder,
        #                 config_dir=self.config_dir,
        #                 dependency=self.dependency)
        # return t2

    # @staticmethod
    # def get_config():
    #     configs = super(EdgeLabelsWorkflow, EdgeLabelsWorkflow).get_config()
    #     configs.update({'block_edge_labels', label_tasks.BlockEdgeLabelsLocal.default_task_config()})
    #     return configs
