import os
import luigi
import json
import z5py
import numpy as np

from ..cluster_tasks import WorkflowBase
from ..relabel import RelabelWorkflow
from ..relabel import find_uniques as unique_tasks
from ..node_labels import NodeLabelWorkflow
from ..features import RegionFeaturesWorkflow
from .. import write as write_tasks

from . import size_filter_blocks as size_filter_tasks
from . import background_size_filter as bg_tasks
from . import filling_size_filter as filling_tasks
from . import filter_blocks as filter_tasks
from . import id_filter as id_tasks
from . import orphan_assignments as orphan_tasks
from . import graph_watershed_assignments as gws_tasks
from . import graph_connected_components as cc_tasks


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
                          config_dir=self.config_dir,
                          input_path=self.input_path,
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
                                  assignment_path=self.output_path,
                                  assignment_key='relabel_size_filter',
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
                      output_path=id_filter_path,
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


class ApplyThreshold(luigi.Task):
    feature_path = luigi.Parameter()
    feature_key = luigi.Parameter()
    out_path = luigi.Parameter()
    threshold = luigi.FloatParameter()
    threshold_mode = luigi.Parameter(default='less')
    dependency = luigi.TaskParameter()

    threshold_modes = ('less', 'greater', 'equal')

    def requires(self):
        return self.dependency

    def run(self):
        f = z5py.File(self.feature_path)
        ds = f[self.feature_key]
        feats = ds[:]

        assert self.threshold_mode in self.threshold_modes
        if self.threshold_mode == 'less':
            filter_ids = feats < self.threshold
        elif self.threshold_mode == 'greater':
            filter_ids = feats > self.threshold
        elif self.threshold_mode == 'equal':
            filter_ids = feats == self.threshold

        filter_ids = np.where(filter_ids)[0].tolist()
        with open(self.out_path, 'w') as f:
            json.dump(filter_ids, f)

    def output(self):
        return luigi.LocalTarget(self.out_path)


class FilterByThresholdWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    seg_in_path = luigi.Parameter()
    seg_in_key = luigi.Parameter()
    seg_out_path = luigi.Parameter()
    seg_out_key = luigi.Parameter()
    threshold = luigi.FloatParameter()
    relabel = luigi.BoolParameter(default=True)

    def requires(self):
        # calculate the region features
        feat_path = os.path.join(self.tmp_folder, 'reg_feats.n5')
        feat_key = 'feats'
        dep = RegionFeaturesWorkflow(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                                     target=self.target, config_dir=self.config_dir,
                                     input_path=self.input_path, input_key=self.input_key,
                                     labels_path=self.seg_in_path, labels_key=self.seg_in_key,
                                     output_path=feat_path, output_key=feat_key)

        # apply threshold to get the ids to filter out
        id_filter_path = os.path.join(self.tmp_folder, 'filtered_ids.json')
        dep = ApplyThreshold(feature_path=feat_path, feature_key=feat_key,
                             out_path=id_filter_path, threshold=self.threshold,
                             dependency=dep)

        # filter all blocks
        filter_task = getattr(filter_tasks,
                              self._get_task_name('FilterBlocks'))
        dep = filter_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                          dependency=dep, max_jobs=self.max_jobs,
                          input_path=self.seg_in_path, input_key=self.seg_in_key,
                          filter_path=id_filter_path,
                          output_path=self.seg_out_path, output_key=self.seg_out_key)
        if self.relabel:
            dep = RelabelWorkflow(tmp_folder=self.tmp_folder,
                                  max_jobs=self.max_jobs,
                                  config_dir=self.config_dir,
                                  target=self.target,
                                  input_path=self.seg_out_path,
                                  input_key=self.seg_out_key,
                                  assignment_path=self.seg_out_path,
                                  assignment_key='assignments/relabel_filter',
                                  dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(FilterByThresholdWorkflow, FilterByThresholdWorkflow).get_config()
        configs.update({'filter_blocks': filter_tasks.FilterBlocksLocal.default_task_config(),
                        **RegionFeaturesWorkflow.get_config()})
        return configs


class FilterOrphansWorkflow(WorkflowBase):

    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()

    path = luigi.Parameter()
    segmentation_key = luigi.Parameter()
    assignment_key = luigi.Parameter()

    output_path = luigi.Parameter()
    assignment_out_key = luigi.Parameter()
    output_key = luigi.Parameter(default=None)
    relabel = luigi.BoolParameter(default=False)

    def requires(self):
        assert False, "FIXME not debugged yet"
        dep = self.dependency
        orphan_task = getattr(orphan_tasks,
                              self._get_task_name('OrphanAssignments'))
        dep = orphan_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=dep,
                          graph_path=self.graph_path, graph_key=self.graph_key,
                          assignment_path=self.path, assignment_key=self.assignment_key,
                          output_path=self.path, output_key=self.assignment_out_key,
                          relabel=self.relabel)
        if self.output_key is not None:
            write_task = getattr(write_tasks,
                                 self._get_task_name('Write'))
            dep = write_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                             config_dir=self.config_dir, dependency=dep,
                             input_path=self.path, input_key=self.segmentation_key,
                             output_path=self.path, output_key=self.output_key,
                             assignment_path=self.output_path, assignment_key=self.assignment_out_key,
                             identifier='filter-orphans')
        return dep

    @staticmethod
    def get_config():
        configs = super(FilterOrphansWorkflow, FilterOrphansWorkflow).get_config()
        configs.update({'orphan_assignments': orphan_tasks.OrphanAssignmentsLocal.default_task_config(),
                        'write': write_tasks.WriteLocal.default_task_config()})
        return configs


class ConnectedComponentsWorkflow(WorkflowBase):
    problem_path = luigi.Parameter()
    graph_key = luigi.Parameter()

    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()

    output_path = luigi.Parameter()
    assignment_out_key = luigi.Parameter()

    output_key = luigi.Parameter(default='')
    path = luigi.Parameter(default='')
    fragments_key = luigi.Parameter(default='')

    def requires(self):
        cc_task = getattr(cc_tasks,
                          self._get_task_name('GraphConnectedComponents'))
        dep = cc_task(max_jobs=self.max_jobs, tmp_folder=self.tmp_folder,
                      config_dir=self.config_dir,
                      problem_path=self.problem_path,
                      graph_key=self.graph_key,
                      assignment_path=self.assignment_path,
                      assignment_key=self.assignment_key,
                      output_path=self.output_path,
                      output_key=self.assignment_out_key,
                      dependency=self.dependency)
        if self.output_key != '':
            write_task = getattr(write_tasks,
                                 self._get_task_name('Write'))
            assert self.fragments_key != '' and self.path != ''
            dep = write_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                             config_dir=self.config_dir, dependency=dep,
                             input_path=self.path, input_key=self.fragments_key,
                             output_path=self.output_path, output_key=self.output_key,
                             assignment_path=self.output_path,
                             assignment_key=self.assignment_out_key,
                             identifier='graph-connected-components')
        return dep

    @staticmethod
    def get_config():
        configs = super(ConnectedComponentsWorkflow, ConnectedComponentsWorkflow).get_config()
        configs.update({'graph_connected_components': cc_tasks.GraphConnectedComponentsLocal.default_task_config(),
                        'write': write_tasks.WriteLocal.default_task_config()})
        return configs


class SizeFilterAndGraphWatershedWorkflow(WorkflowBase):

    problem_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    features_key = luigi.Parameter()

    #
    path = luigi.Parameter()
    # path to the merged segmentation
    segmentation_key = luigi.Parameter()
    # path to the underlying fragments
    fragments_key = luigi.Parameter(default='')
    # path to the fragment segment assignment
    assignment_key = luigi.Parameter()

    # the size filter threshold
    size_threshold = luigi.IntParameter()
    relabel = luigi.BoolParameter(default=False)
    from_costs = luigi.BoolParameter(default=False)

    output_path = luigi.Parameter()
    assignment_out_key = luigi.Parameter()
    output_key = luigi.Parameter(default='')

    def find_sizes(self, dep):
        # find segemnts that should be merged according to the size filter
        un_task = getattr(unique_tasks,
                          self._get_task_name('FindUniques'))
        dep = un_task(tmp_folder=self.tmp_folder,
                      max_jobs=self.max_jobs,
                      config_dir=self.config_dir,
                      input_path=self.path,
                      input_key=self.segmentation_key,
                      return_counts=True,
                      dependency=dep)
        sf_task = getattr(size_filter_tasks,
                          self._get_task_name('SizeFilterBlocks'))
        dep = sf_task(tmp_folder=self.tmp_folder,
                      max_jobs=self.max_jobs,
                      config_dir=self.config_dir,
                      input_path=self.path,
                      input_key=self.segmentation_key,
                      size_threshold=self.size_threshold,
                      dependency=dep)
        return dep

    def requires(self):
        dep = self.dependency

        # find the sizes for all segments
        dep = self.find_sizes(dep)

        # run graph watershed to merge in all small segments
        filter_path = os.path.join(self.tmp_folder, 'discard_ids.npy')
        gws_task = getattr(gws_tasks,
                           self._get_task_name('GraphWatershedAssignments'))
        dep = gws_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                       config_dir=self.config_dir, dependency=dep,
                       problem_path=self.problem_path, graph_key=self.graph_key,
                       features_key=self.features_key,
                       assignment_path=self.path, assignment_key=self.assignment_key,
                       output_path=self.output_path, output_key=self.assignment_out_key,
                       filter_nodes_path=filter_path,
                       relabel=self.relabel, from_costs=self.from_costs)

        if self.output_key != '':
            assert self.fragments_key != ''
            write_task = getattr(write_tasks,
                                 self._get_task_name('Write'))
            dep = write_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                             config_dir=self.config_dir, dependency=dep,
                             input_path=self.path, input_key=self.fragments_key,
                             output_path=self.output_path, output_key=self.output_key,
                             assignment_path=self.output_path, assignment_key=self.assignment_out_key,
                             identifier='size-filter-graph-ws')
        return dep

    @staticmethod
    def get_config():
        configs = super(SizeFilterAndGraphWatershedWorkflow,
                        SizeFilterAndGraphWatershedWorkflow).get_config()
        configs.update({'size_filter_blocks': size_filter_tasks.SizeFilterBlocksLocal.default_task_config(),
                        'graph_watershed_assignments':
                        gws_tasks.GraphWatershedAssignmentsLocal.default_task_config(),
                        'write': write_tasks.WriteLocal.default_task_config()})
        return configs
