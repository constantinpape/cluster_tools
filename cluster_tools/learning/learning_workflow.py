import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
# TODO Region features
from ..features import EdgeFeaturesWorkflow
from ..graph import GraphWorkflow
from ..node_labels import NodeLabelWorkflow
from . import edge_labels as label_tasks
from . import learn_rf as learn_tasks


class LearningWorkflow(WorkflowBase):
    input_dict = luigi.DictParameter()
    labels_dict = luigi.DictParameter()
    groundtruth_dict = luigi.DictParameter()
    output_path = luigi.Parameter()

    def _check_input(self):
        assert self.input_dict.keys() == self.labels_dict.keys()
        assert self.input_dict.keys() == self.groundtruth_dict.keys()

    def requires(self):
        self._check_input()
        n_scales = 1

        edge_labels_dict = {}
        features_dict = {}

        try:
            os.mkdir(self.tmp_folder)
        except OSError:
            pass

        prev_dep = self.dependency
        for key, input_path in self.input_dict.items():
            labels_path = self.labels_dict[key]
            gt_path = self.groundtruth_dict[key]

            # we need different tmp folders for each input dataset
            tmp_folder = os.path.join(self.tmp_folder, key)

            graph_out = os.path.join(tmp_folder, 'graph.n5')
            graph_task = GraphWorkflow(tmp_folder=tmp_folder,
                                       max_jobs=self.max_jobs,
                                       config_dir=self.config_dir,
                                       target=self.target,
                                       dependency=prev_dep,
                                       input_path=labels_path[0],
                                       input_key=labels_path[1],
                                       graph_path=graph_out,
                                       output_key='graph',
                                       n_scales=n_scales)

            features_out = os.path.join(tmp_folder, 'features.n5')
            feat_task = EdgeFeaturesWorkflow(tmp_folder=tmp_folder,
                                             max_jobs=self.max_jobs,
                                             config_dir=self.config_dir,
                                             dependency=graph_task,
                                             target=self.target,
                                             input_path=input_path[0],
                                             input_key=input_path[1],
                                             labels_path=labels_path[0],
                                             labels_key=labels_path[1],
                                             graph_path=graph_out,
                                             graph_key='graph',
                                             output_path=features_out,
                                             output_key='features')
            features_dict[key] = (features_out, 'features')

            node_labels_out = os.path.join(tmp_folder, 'gt_node_labels.n5')
            node_labels_task = NodeLabelWorkflow(tmp_folder=tmp_folder,
                                            max_jobs=self.max_jobs,
                                            config_dir=self.config_dir,
                                            target=self.target,
                                            dependency=feat_task,
                                            ws_path=labels_path[0],
                                            ws_key=labels_path[1],
                                            input_path=gt_path[0],
                                            input_key=gt_path[1],
                                            output_path=node_labels_out,
                                            output_key='node_labels')

            edge_labels_out = os.path.join(tmp_folder, 'edge_labels.n5')
            lt = getattr(label_tasks,
                         self._get_task_name('EdgeLabels'))
            label_task = lt(tmp_folder=tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            dependency=node_labels_task,
                            graph_path=graph_out,
                            graph_key='graph',
                            node_labels_path=node_labels_out,
                            node_labels_key='node_labels',
                            output_path=edge_labels_out,
                            output_key='edge_labels')
            prev_dep = label_task
            edge_labels_dict[key] = (edge_labels_out, 'edge_labels')

        learn_task = getattr(learn_tasks,
                             self._get_task_name('LearnRF'))
        rf_task = learn_task(tmp_folder=self.tmp_folder,
                             config_dir=self.config_dir,
                             max_jobs=self.max_jobs,
                             features_dict=features_dict,
                             labels_dict=edge_labels_dict,
                             output_path=self.output_path,
                             dependency=prev_dep)
        return rf_task

    @staticmethod
    def get_config():
        configs = {'learn_rf': learn_tasks.LearnRFLocal.default_task_config(),
                   'edge_labels': label_tasks.EdgeLabelsLocal.default_task_config(),
                   **GraphWorkflow.get_config(),
                   **EdgeFeaturesWorkflow.get_config()}
        return configs
