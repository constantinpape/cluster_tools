import os
import json
import luigi

from .. cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from ..node_labels import NodeLabelWorkflow
from . import sparse_lifted_neighborhood as nh_tasks
from . import costs_from_node_labels as cost_tasks


class LiftedFeaturesFromNodeLabelsWorkflow(WorkflowBase):
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    prefix = luigi.Parameter()
    nh_graph_depth = luigi.IntParameter(default=4)

    def requires(self):
        # 1.) get the node labels from overlapping labels from `labels_path`
        # with the over-segmentation in `ws_path`
        labels_key = 'node_overlaps/%s' % self.prefix
        dep = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                max_jobs=self.max_jobs, dependency=self.dependency,
                                target=self.target,
                                ws_path=self.ws_path, ws_key=self.ws_key,
                                input_path=self.labels_path, input_key=self.labels_key,
                                prefix=self.prefix, output_path=self.output_path,
                                output_key=labels_key, max_overlap=True)
        # 2.) find the sparse lifted neighborhood based on the node overlaps
        # and the neighborhood graph depth
        nh_task = getattr(nh_tasks,
                          self._get_task_name('SparseLiftedNeighborhood'))
        nh_key = 'lifted_neighborhoods/%s' % self.prefix
        dep = nh_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                      max_jobs=self.max_jobs, dependency=dep,
                      graph_path=self.graph_path, graph_key=self.graph_key,
                      node_label_path=self.output_path, node_label_key=labels_key,
                      output_path=self.output_path, output_key=nh_key,
                      nh_graph_depth=self.nh_graph_depth,
                      prefix=self.prefix)

        # 3.) find the lifted features based on neighborhood and node labels
        cost_task = getattr(cost_tasks,
                            self._get_task_name('CostsFromNodeLabels'))
        feat_key = self.output_key
        dep = cost_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                        max_jobs=self.max_jobs, dependency=dep,
                        nh_path=self.output_path, nh_key=nh_key,
                        node_label_path=self.output_path, node_label_key=labels_key,
                        output_path=self.output_path, output_key=feat_key,
                        prefix=self.prefix)
        return dep

    @staticmethod
    def get_config():
        configs = super(LiftedFeaturesFromNodeLabelsWorkflow,
                        LiftedFeaturesFromNodeLabelsWorkflow).get_config()
        configs.update({**NodeLabelWorkflow.get_config(),
                        'sparse_lifted_neighborhood':
                        nh_tasks.SparseLiftedNeighborhoodLocal.default_task_config(),
                        'features_from_node_labels':
                        cost_tasks.CostsFromNodeLabelsLocal.default_task_config()})
        return configs
