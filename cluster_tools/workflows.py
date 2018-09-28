import os
import json
import luigi

from .cluster_tasks import WorkflowBase
from .watershed import WatershedWorkflow
from .graph import GraphWorkflow
# TODO more features and options to choose which features to choose
from .features import EdgeFeaturesWorkflow
from .costs import EdgeCostsWorkflow
from .multicut import MulticutWorkflow
from .decomposition_multicut import DecompositionWorkflow
from . import write as write_tasks


class MulticutSegmentationWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    # where to save the watersheds
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    # where to save the graph
    graph_path = luigi.Parameter()
    # where to save the features
    features_path = luigi.Parameter()
    # where to save the costs
    costs_path = luigi.Parameter()
    # where to save the node labels
    node_labels_path = luigi.Parameter()
    node_labels_key = luigi.Parameter()
    # where to save the intermediate multicut problems
    problem_path = luigi.Parameter()
    # where to save the resulting segmentation
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # number of scales
    n_scales = luigi.IntParameter()
    # optional path to mask
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')
    # number of jobs used in feature merging
    max_jobs_merge_features = luigi.IntParameter(default=1)
    # number of jobs used for sub multicuts
    max_jobs_multicut = luigi.IntParameter(default=1)
    # use decomposer workflow
    use_demposition_multicut = luigi.BoolParameter(default=False)
    # path to random forest (if available)
    rf_path = luigi.Parameter(default='')
    # TODO list to skip jobs

    def _get_mc_wf(self):
        # hard-coded keys
        graph_key = 'graph'
        features_key = 'features'
        costs_key = 'costs'
        if self.use_decomposition_multicut:
            mc_wf = DecompositionWorkflow(tmp_folder=self.tmp_folder,
                                          max_jobs=self.max_jobs_multicut,
                                          config_dir=self.config_dir,
                                          target=self.target,
                                          dependency=costs_wf,
                                          graph_path=self.graph_path,
                                          graph_key=graph_key,
                                          costs_path=self.costs_path,
                                          costs_key=costs_key,
                                          output_path=self.node_labels_path,
                                          output_key=self.node_labels_key,
                                          decomposition_path=self.output_path)
        else:
            mc_wf = MulticutWorkflow(tmp_folder=self.tmp_folder,
                                     max_jobs=self.max_jobs_multicut,
                                     config_dir=self.config_dir,
                                     target=self.target,
                                     dependency=costs_wf,
                                     graph_path=self.graph_path,
                                     graph_key=graph_key,
                                     costs_path=self.costs_path,
                                     costs_key=costs_key,
                                     n_scales=self.n_scales,
                                     output_path=self.node_labels_path,
                                     output_key=self.node_labels_key,
                                     merged_problem_path=self.problem_path)
        return mc_wf

    # TODO implement mechanism to skip existing dependencies
    def requires(self):
        # hard-coded keys
        graph_key = 'graph'
        features_key = 'features'
        costs_key = 'costs'
        ws_wf = WatershedWorkflow(tmp_folder=self.tmp_folder,
                                  max_jobs=self.max_jobs,
                                  config_dir=self.config_dir,
                                  target=self.target,
                                  dependency=self.dependency,
                                  input_path=self.input_path,
                                  input_key=self.input_key,
                                  output_path=self.ws_path,
                                  output_key=self.ws_key,
                                  mask_path=self.mask_path,
                                  mask_key=self.mask_key)
        # return ws_wf
        # TODO in the current implementation, we can only compute the
        # graph with n_scales=1, otherwise we will clash with the
        # multicut merged graphs
        graph_wf = GraphWorkflow(tmp_folder=self.tmp_folder,
                                 max_jobs=self.max_jobs,
                                 config_dir=self.config_dir,
                                 target=self.target,
                                 dependency=ws_wf,
                                 # dependency=self.dependency,
                                 input_path=self.ws_path,
                                 input_key=self.ws_key,
                                 graph_path=self.graph_path,
                                 n_scales=1)
        # TODO add options to choose which features to use
        features_wf = EdgeFeaturesWorkflow(tmp_folder=self.tmp_folder,
                                           max_jobs=self.max_jobs,
                                           config_dir=self.config_dir,
                                           target=self.target,
                                           dependency=graph_wf,
                                           input_path=self.input_path,
                                           input_key=self.input_key,
                                           labels_path=self.ws_path,
                                           labels_key=self.ws_key,
                                           graph_path=self.graph_path,
                                           graph_key=graph_key,
                                           output_path=self.features_path,
                                           output_key=features_key,
                                           max_jobs_merge=self.max_jobs_merge_features)
        costs_wf = EdgeCostsWorkflow(tmp_folder=self.tmp_folder,
                                     max_jobs=self.max_jobs,
                                     config_dir=self.config_dir,
                                     target=self.target,
                                     dependency=features_wf,
                                     features_path=self.features_path,
                                     features_key=features_key,
                                     output_path=self.costs_path,
                                     output_key=costs_key,
                                     rf_path=self.rf_path)
        mc_wf = self._get_mc_wf()
        write_task = getattr(write_tasks,
                             self._get_task_name('Write'))
        t = write_task(tmp_folder=self.tmp_folder,
                       max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       dependency=mc_wf,
                       input_path=self.ws_path,
                       input_key=self.ws_key,
                       output_path=self.output_path,
                       output_key=self.output_key,
                       identifier='multicut')
        return t

    @staticmethod
    def get_config():
        config = {**WatershedWorkflow.get_config(),
                  **GraphWorkflow.get_config(),
                  **EdgeFeaturesWorkflow.get_config(),
                  **EdgeCostsWorkflow.get_config()}
        if self.use_decomposition_multicut:
            config.update(**DecompositionWorkflow.get_config())
        else:
            config.update(**MulticutWorkflow.get_config())
        return config
