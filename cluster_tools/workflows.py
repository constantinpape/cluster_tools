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
from .debugging import CheckSubGraphsWorkflow
from . import write as write_tasks


class MulticutSegmentationWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    # where to save the watersheds
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    # where to save the multicut problems
    problem_path = luigi.Parameter()
    # where to save the node labels
    node_labels_path = luigi.Parameter()
    node_labels_key = luigi.Parameter()
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
    skip_ws = luigi.BoolParameter(default=False)
    # path to random forest (if available)
    rf_path = luigi.Parameter(default='')
    # node label dict: dictionary for additional node labels used in costs
    node_label_dict = luigi.DictParameter(default={})
    # run some sanity checks for sub-results
    sanity_checks = luigi.BoolParameter(default=False)
    # TODO list to skip jobs

    def _get_mc_wf(self, dep):
        # hard-coded keys
        mc_wf = MulticutWorkflow(tmp_folder=self.tmp_folder,
                                 max_jobs=self.max_jobs_multicut,
                                 config_dir=self.config_dir,
                                 target=self.target,
                                 dependency=dep,
                                 problem_path=self.problem_path,
                                 n_scales=self.n_scales,
                                 assignment_path=self.node_labels_path,
                                 assignment_key=self.node_labels_key)
        return mc_wf

    # TODO implement mechanism to skip existing dependencies
    def requires(self):
        # hard-coded keys
        graph_key = 's0/graph'
        features_key = 'features'
        costs_key = 's0/costs'
        if self.skip_ws:
            assert os.path.exists(os.path.join(self.ws_path, self.ws_key)), "%s:%s" % (self.ws_path,
                                                                                       self.ws_key)
            dep = self.dependency
        else:
            dep = WatershedWorkflow(tmp_folder=self.tmp_folder,
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
        # TODO in the current implementation, we can only compute the
        # graph with n_scales=1, otherwise we will clash with the
        # multicut merged graphs
        dep = GraphWorkflow(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            target=self.target,
                            dependency=dep,
                            input_path=self.ws_path,
                            input_key=self.ws_key,
                            graph_path=self.problem_path,
                            output_key=graph_key,
                            n_scales=1)
        if self.sanity_checks:
            graph_block_prefix = os.path.join(self.problem_path,
                                              's0', 'sub_graphs', 'block_')
            dep = CheckSubGraphsWorkflow(tmp_folder=self.tmp_folder,
                                         max_jobs=self.max_jobs,
                                         config_dir=self.config_dir,
                                         target=self.target,
                                         ws_path=self.ws_path,
                                         ws_key=self.ws_key,
                                         graph_block_prefix=graph_block_prefix,
                                         dependency=dep)
        # TODO add options to choose which features to use
        dep = EdgeFeaturesWorkflow(tmp_folder=self.tmp_folder,
                                   max_jobs=self.max_jobs,
                                   config_dir=self.config_dir,
                                   target=self.target,
                                   dependency=dep,
                                   input_path=self.input_path,
                                   input_key=self.input_key,
                                   labels_path=self.ws_path,
                                   labels_key=self.ws_key,
                                   graph_path=self.problem_path,
                                   graph_key=graph_key,
                                   output_path=self.problem_path,
                                   output_key=features_key,
                                   max_jobs_merge=self.max_jobs_merge_features)
        dep = EdgeCostsWorkflow(tmp_folder=self.tmp_folder,
                                max_jobs=self.max_jobs,
                                config_dir=self.config_dir,
                                target=self.target,
                                dependency=dep,
                                features_path=self.problem_path,
                                features_key=features_key,
                                output_path=self.problem_path,
                                output_key=costs_key,
                                node_label_dict=self.node_label_dict,
                                rf_path=self.rf_path)
        dep = self._get_mc_wf(dep)
        write_task = getattr(write_tasks, self._get_task_name('Write'))
        dep = write_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         dependency=dep,
                         input_path=self.ws_path,
                         input_key=self.ws_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         assignment_path=self.node_labels_path,
                         assignment_key=self.node_labels_key,
                         identifier='multicut')
        return dep

    @staticmethod
    def get_config():
        config = {**WatershedWorkflow.get_config(),
                  **GraphWorkflow.get_config(),
                  **EdgeFeaturesWorkflow.get_config(),
                  **EdgeCostsWorkflow.get_config(),
                  **MulticutWorkflow.get_config()}
        return config
