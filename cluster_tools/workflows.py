import os
import luigi

from .cluster_tasks import WorkflowBase
from .watershed import WatershedWorkflow
from .graph import GraphWorkflow

# TODO more features and options to choose which features to use
from .features import EdgeFeaturesWorkflow
from .costs import EdgeCostsWorkflow

from .multicut import MulticutWorkflow
from .lifted_features import LiftedFeaturesFromNodeLabelsWorkflow
from .lifted_multicut import LiftedMulticutWorkflow

from .debugging import CheckSubGraphsWorkflow
from . import write as write_tasks

#
from .agglomerative_clustering import agglomerative_clustering as agglomerate_tasks
from .stitching import stitching_multicut as stitch_mc_tasks
from .stitching import StitchingAssignmentsWorkflow


# TODO add options to choose which features to use
# NOTE in the current implementation, we can only compute the
# graph with n_scales=1, otherwise we will clash with the multicut merged graphs
class ProblemWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    problem_path = luigi.Parameter()

    # optional params for costs
    rf_path = luigi.Parameter(default='')
    node_label_dict = luigi.DictParameter(default={})

    max_jobs_merge = luigi.IntParameter(default=None)
    # do we compte costs
    compute_costs = luigi.BoolParameter(default=True)
    # do we run sanity checks ?
    sanity_checks = luigi.BoolParameter(default=False)

    # hard-coded keys
    graph_key = 's0/graph'
    features_key = 'features'
    costs_key = 's0/costs'

    def requires(self):
        dep = GraphWorkflow(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            target=self.target,
                            dependency=self.dependency,
                            input_path=self.ws_path,
                            input_key=self.ws_key,
                            graph_path=self.problem_path,
                            output_key=self.graph_key,
                            n_scales=1)
        # sanity check the subgraph
        if self.sanity_checks:
            subgraph_key = 's0/sub_graphs'
            dep = CheckSubGraphsWorkflow(tmp_folder=self.tmp_folder,
                                         max_jobs=self.max_jobs,
                                         config_dir=self.config_dir,
                                         target=self.target,
                                         ws_path=self.ws_path,
                                         ws_key=self.ws_key,
                                         graph_path=self.problem_path,
                                         subgraph_key=subgraph_key,
                                         dependency=dep)
        max_jobs_merge = self.max_jobs if self.max_jobs_merge is None else self.max_jobs_merge
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
                                   graph_key=self.graph_key,
                                   output_path=self.problem_path,
                                   output_key=self.features_key,
                                   max_jobs_merge=max_jobs_merge)
        if self.compute_costs:
            dep = EdgeCostsWorkflow(tmp_folder=self.tmp_folder,
                                    max_jobs=self.max_jobs,
                                    config_dir=self.config_dir,
                                    target=self.target,
                                    dependency=dep,
                                    features_path=self.problem_path,
                                    features_key=self.features_key,
                                    output_path=self.problem_path,
                                    output_key=self.costs_key,
                                    node_label_dict=self.node_label_dict,
                                    seg_path=self.ws_path, seg_key=self.ws_key,
                                    rf_path=self.rf_path)
        return dep

    @staticmethod
    def get_config():
        config = {**GraphWorkflow.get_config(),
                  **EdgeFeaturesWorkflow.get_config(),
                  **EdgeCostsWorkflow.get_config()}
        return config


class SegmentationWorkflowBase(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    # where to save the watersheds
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    # where to save the problem (graph, edge_features etc)
    problem_path = luigi.Parameter()
    # where to save the node labels
    node_labels_key = luigi.Parameter()
    # where to save the resulting segmentation
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # optional path to mask
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    # optional path for random forest used for cost computation
    rf_path = luigi.Parameter(default='')
    # node label dict: dictionary for additional node labels used in costs
    node_label_dict = luigi.DictParameter(default={})

    # number of jobs used for merge tasks
    max_jobs_merge = luigi.IntParameter(default=None)
    # skip watershed (watershed volume must already be preset)
    skip_ws = luigi.BoolParameter(default=False)
    # run agglomeration immediately after watersheds
    agglomerate_ws = luigi.BoolParameter(default=False)
    # run two-pass watershed
    two_pass_ws = luigi.BoolParameter(default=False)
    # run some sanity checks for intermediate results
    sanity_checks = luigi.BoolParameter(default=False)

    # hard-coded keys
    graph_key = 's0/graph'
    features_key = 'features'
    costs_key = 's0/costs'

    def _watershed_tasks(self):
        if self.skip_ws:
            assert os.path.exists(os.path.join(self.ws_path, self.ws_key)), "%s:%s" % (self.ws_path,
                                                                                       self.ws_key)
            return self.dependency
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
                                    mask_key=self.mask_key,
                                    two_pass=self.two_pass_ws,
                                    agglomeration=self.agglomerate_ws)
            return dep

    def _problem_tasks(self, dep, compute_costs):
        dep = ProblemWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                              max_jobs=self.max_jobs, target=self.target, dependency=dep,
                              input_path=self.input_path, input_key=self.input_key,
                              ws_path=self.ws_path, ws_key=self.ws_key,
                              problem_path=self.problem_path, rf_path=self.rf_path,
                              node_label_dict=self.node_label_dict,
                              max_jobs_merge=self.max_jobs_merge,
                              compute_costs=compute_costs, sanity_checks=self.sanity_checks)
        return dep

    def _write_tasks(self, dep, identifier):
        if self.output_key == '':
            return dep
        write_task = getattr(write_tasks, self._get_task_name('Write'))
        dep = write_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         dependency=dep,
                         input_path=self.ws_path,
                         input_key=self.ws_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         assignment_path=self.output_path,
                         assignment_key=self.node_labels_key,
                         identifier=identifier)
        return dep

    @staticmethod
    def get_config():
        config = {**WatershedWorkflow.get_config(), **ProblemWorkflow.get_config()}
        return config


class MulticutSegmentationWorkflow(SegmentationWorkflowBase):
    # number of jobs used for sub multicuts
    max_jobs_multicut = luigi.IntParameter(default=1)
    # number of scales
    n_scales = luigi.IntParameter(default=1)

    def _multicut_tasks(self, dep):
        dep = MulticutWorkflow(tmp_folder=self.tmp_folder,
                               max_jobs=self.max_jobs_multicut,
                               config_dir=self.config_dir,
                               target=self.target,
                               dependency=dep,
                               problem_path=self.problem_path,
                               n_scales=self.n_scales,
                               assignment_path=self.output_path,
                               assignment_key=self.node_labels_key)
        return dep

    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=True)
        dep = self._multicut_tasks(dep)
        dep = self._write_tasks(dep, 'multicut')
        return dep

    @staticmethod
    def get_config():
        config = super(MulticutSegmentationWorkflow, MulticutSegmentationWorkflow).get_config()
        config.update(MulticutWorkflow.get_config())
        return config


class LiftedMulticutSegmentationWorkflow(SegmentationWorkflowBase):
    # number of jobs used for sub multicuts
    max_jobs_multicut = luigi.IntParameter(default=1)
    # number of scales
    n_scales = luigi.IntParameter()

    # node labels for lifted milticut
    # set to '' in order to skip sparse lifted edge computation
    # in this case, pre-computed lifted edges and costs need to be provided
    lifted_labels_path = luigi.Parameter()
    lifted_labels_key = luigi.Parameter()
    # prefix to distinguish task and log-names
    lifted_prefix = luigi.Parameter()
    # graph depth for lifted neighborhood
    nh_graph_depth = luigi.IntParameter(default=4)
    # ignore labels:
    # node_ignore_label: this label will be ignored when inserting lifted
    # edges into the lifted neighborhood (usually 0 for sparse lifted edges)
    node_ignore_label = luigi.IntParameter(default=0)
    # label_ignore_label: this label will be ignored when mapping the lifted_labels
    # (lifted_labels_path:lifted_labels_key) to nodes for constructing sparse
    # lifted edges (usually None -> no label is ignored)
    label_ignore_label = luigi.Parameter(default=None)
    mode = luigi.Parameter(default='all')
    label_overlap_threshold = luigi.Parameter(default=None)
    # clear labels
    clear_labels_path = luigi.Parameter(default=None)
    clear_labels_key = luigi.Parameter(default=None)

    def _lifted_problem_tasks(self, dep):
        nh_key = 's0/lifted_nh_%s' % self.lifted_prefix
        feat_key = 's0/lifted_costs_%s' % self.lifted_prefix
        dep = LiftedFeaturesFromNodeLabelsWorkflow(tmp_folder=self.tmp_folder,
                                                   max_jobs=self.max_jobs,
                                                   config_dir=self.config_dir,
                                                   target=self.target,
                                                   dependency=dep,
                                                   ws_path=self.ws_path,
                                                   ws_key=self.ws_key,
                                                   labels_path=self.lifted_labels_path,
                                                   labels_key=self.lifted_labels_key,
                                                   output_path=self.problem_path,
                                                   nh_out_key=nh_key,
                                                   feat_out_key=feat_key,
                                                   graph_path=self.problem_path,
                                                   graph_key=self.graph_key,
                                                   prefix=self.lifted_prefix,
                                                   nh_graph_depth=self.nh_graph_depth,
                                                   ignore_label=self.node_ignore_label,
                                                   label_ignore_label=self.label_ignore_label,
                                                   label_overlap_threshold=self.label_overlap_threshold,
                                                   clear_labels_path=self.clear_labels_path,
                                                   clear_labels_key=self.clear_labels_key,
                                                   mode=self.mode)
        return dep

    def _lifted_multicut_tasks(self, dep):
        dep = LiftedMulticutWorkflow(tmp_folder=self.tmp_folder,
                                     max_jobs=self.max_jobs_multicut,
                                     config_dir=self.config_dir,
                                     target=self.target,
                                     dependency=dep,
                                     problem_path=self.problem_path,
                                     n_scales=self.n_scales,
                                     assignment_path=self.output_path,
                                     assignment_key=self.node_labels_key,
                                     lifted_prefix=self.lifted_prefix)
        return dep

    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=True)
        # enable splitting the lifted problem calculation
        # to allow for precomputed costs
        if self.lifted_labels_path != '':
            assert self.lifted_labels_key != ''
            dep = self._lifted_problem_tasks(dep)
        dep = self._lifted_multicut_tasks(dep)
        dep = self._write_tasks(dep, 'lifted_multicut')
        return dep

    @staticmethod
    def get_config():
        config = super(LiftedMulticutSegmentationWorkflow,
                       LiftedMulticutSegmentationWorkflow).get_config()
        config.update({**LiftedFeaturesFromNodeLabelsWorkflow.get_config(),
                       **LiftedMulticutWorkflow.get_config()})
        return config


# TODO support vanilla agglomerative clustering
class AgglomerativeClusteringWorkflow(SegmentationWorkflowBase):
    threshold = luigi.FloatParameter()

    def _agglomerate_task(self, dep):
        agglomerate_task = getattr(agglomerate_tasks,
                                   self._get_task_name('AgglomerativeClustering'))
        dep = agglomerate_task(tmp_folder=self.tmp_folder,
                               max_jobs=self.max_jobs,
                               config_dir=self.config_dir,
                               problem_path=self.problem_path,
                               assignment_path=self.output_path,
                               assignment_key=self.node_labels_key,
                               features_path=self.problem_path,
                               features_key=self.features_key,
                               threshold=self.threshold,
                               dependency=dep)
        return dep

    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=False)
        dep = self._agglomerate_task(dep)
        dep = self._write_tasks(dep, 'agglomerative_clustering')
        return dep

    @staticmethod
    def get_config():
        configs = super(AgglomerativeClusteringWorkflow,
                        AgglomerativeClusteringWorkflow).get_config()
        configs.update({'agglomerative_clustering':
                        agglomerate_tasks.AgglomerativeClusteringLocal.default_task_config()})
        return configs


class SimpleStitchingWorkflow(MulticutSegmentationWorkflow):
    edge_size_threshold = luigi.IntParameter()

    def _simple_stitcher(self, dep):
        dep = StitchingAssignmentsWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                           max_job=self.max_jobs, target=self.target, dependency=dep,
                                           problem_path=self.problem_path,
                                           labels_path=self.ws_path, labels_key=self.ws_key,
                                           assignment_path=self.output_path, assignments_key=self.node_labels_key,
                                           features_key=self.features_key, graph_key=self.graph_key,
                                           edge_size_threshold=self.edge_size_threshold)
        return dep

    def requires(self):
        dep = self._watershed_tasks()
        dep = self._problem_tasks(dep, compute_costs=False)
        dep = self._simple_stitcher(dep)
        dep = self._write_tasks(dep, 'simple_stitching')
        return dep

    @staticmethod
    def get_config():
        configs = super(SimpleStitchingWorkflow,
                        SimpleStitchingWorkflow).get_config()
        configs.update(StitchingAssignmentsWorkflow.get_config())
        return configs


class MulticutStitchingWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    problem_path = luigi.Parameter()
    output_path = luigi.Parameter(default=None)
    output_key = luigi.Parameter(default=None)

    def requires(self):
        # 1.) compute the multicut problem
        dep = ProblemWorkflow(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                              target=self.target, config_dir=self.config_dir,
                              input_path=self.input_path, input_key=self.input_key,
                              ws_path=self.labels_path, ws_key=self.labels_key,
                              problem_path=self.problem_path, compute_costs=False,
                              dependency=self.dependency)

        # 2.) compute the stitch edges
        edge_key = 'stitch_edges'
        graph_key = 's0/graph'
        feat_key = 'features'
        dep = StitchingAssignmentsWorkflow(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                                           target=self.target, config_dir=self.config_dir,
                                           problem_path=self.problem_path,
                                           labels_path=self.labels_path, labels_key=self.labels_key,
                                           assignments_path=self.problem_path, assignments_key=edge_key,
                                           features_key=feat_key, graph_key=graph_key, serialize_edges=True,
                                           dependency=dep)

        # 3.) solve the stitching mc problem
        stitch_mc_task = getattr(stitch_mc_tasks,
                                 self._get_task_name('StitchingMulticut'))
        dep = stitch_mc_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs, config_dir=self.config_dir,
                             problem_path=self.problem_path, graph_key=graph_key,
                             features_key=feat_key, edge_key=edge_key,
                             output_path=self.assignment_path, output_key=self.assignment_key,
                             dependency=dep)

        # 4.) write the output segmentation
        if self.output_path is not None:
            write_task = getattr(write_tasks, self._get_task_name('Write'))
            dep = write_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             input_path=self.labels_path, input_key=self.labels_key,
                             output_path=self.output_path, output_key=self.output_key,
                             assignment_path=self.assignment_path,
                             assignment_key=self.assignment_key,
                             dependency=dep, identifier='stitch-mc')
        return dep

    @staticmethod
    def get_config():
        configs = super(MulticutStitchingWorkflow,
                        MulticutStitchingWorkflow).get_config()
        configs.update({'write': write_tasks.WriteLocal.default_task_config(),
                        'stitching_multicut': stitch_mc_tasks.StitchingMulticutLocal.default_task_config(),
                        **ProblemWorkflow.get_config(),
                        **StitchingAssignmentsWorkflow.get_config()})
        return configs
