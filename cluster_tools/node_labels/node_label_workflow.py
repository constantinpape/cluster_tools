import os
import json
import luigi

from .. cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from . import block_node_labels as label_tasks
from . import merge_node_labels as merge_tasks


class NodeLabelWorkflow(WorkflowBase):
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    prefix = luigi.Parameter(default='')
    max_overlap = luigi.BoolParameter(default=True)

    def requires(self):
        label_task = getattr(label_tasks,
                             self._get_task_name('BlockNodeLabels'))
        tmp_key = 'label_overlaps_%s' % self.prefix
        dep = label_task(max_jobs=self.max_jobs,
                         tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         dependency=self.dependency,
                         ws_path=self.ws_path,
                         ws_key=self.ws_key,
                         input_path=self.input_path,
                         input_key=self.input_key,
                         output_path=self.output_path,
                         output_key=tmp_key)
        merge_task = getattr(merge_tasks,
                             self._get_task_name('MergeNodeLabels'))
        try:
            with vu.file_reader(self.ws_path) as f:
                number_of_labels = f[self.ws_key].attrs['maxId'] + 1
        except KeyError as e:
            msg = "Expect attribute maxId in %s:%s" % (self.ws_path, self.ws_key)
            raise KeyError(msg)
        dep = merge_task(input_path=self.output_path,
                         input_key=tmp_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         max_overlap=self.max_overlap,
                         max_jobs=self.max_jobs,
                         tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         number_of_labels=number_of_labels,
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(NodeLabelWorkflow, NodeLabelWorkflow).get_config()
        configs.update({'block_node_labels':
                        label_tasks.BlockNodeLabelsLocal.default_task_config(),
                        'merge_node_labels':
                        merge_tasks.MergeNodeLabelsLocal.default_task_config()})
        return configs
