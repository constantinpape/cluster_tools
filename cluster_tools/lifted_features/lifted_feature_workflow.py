import luigi
import os
import numpy as np
import z5py
import nifty.distributed as ndist

from .. cluster_tasks import WorkflowBase
from ..node_labels import NodeLabelWorkflow
from . import sparse_lifted_neighborhood as nh_tasks
from . import costs_from_node_labels as cost_tasks
from . import clear_lifted_edges_from_labels as clear_tasks


class NodeLabelsWithThreshold(luigi.Task):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    threshold = luigi.FloatParameter()
    ignore_label = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def normalize_overlap_dict(self, label_overlap_dict):
        sums = {label_id: sum(overlaps.values())
                for label_id, overlaps in label_overlap_dict.items()}
        if self.ignore_label is not None:
            label_overlap_dict = {label_id: {ovlp_id: count / sums[label_id]
                                             for ovlp_id, count in overlaps.items() if ovlp_id != self.ignore_label}
                                  for label_id, overlaps in label_overlap_dict.items()}
        else:
            label_overlap_dict = {label_id: {ovlp_id: count / sums[label_id]
                                             for ovlp_id, count in overlaps.items()}
                                  for label_id, overlaps in label_overlap_dict.items()}
        return label_overlap_dict

    def run(self):
        ds = z5py.File(self.input_path)[self.input_key]
        n_chunks = ds.number_of_chunks
        overlaps = [ndist.deserializeOverlapChunk(ds.path, (chunk_id,))[0]
                    for chunk_id in range(n_chunks)]
        overlaps = {label_id: overlaps
                    for chunk_overlaps in overlaps
                    for label_id, overlaps in chunk_overlaps.items()}
        overlaps = self.normalize_overlap_dict(overlaps)

        n_labels = max(overlaps.keys())
        node_labels = []
        for label_id in range(n_labels):
            ovlps = overlaps.get(label_id, {})
            if not ovlps:
                node_labels.append(self.ignore_label)
                continue

            max_id = -1
            max_ovlp = 0

            for ovlp_id, ovlp in ovlps.items():
                if ovlp > max_ovlp:
                    max_id = ovlp_id
                    max_ovlp = ovlp

            if max_ovlp < self.threshold:
                max_id = self.ignore_label
            node_labels.append(max_id)

        node_labels = np.array(node_labels, dtype='uint64')

        with z5py.File(self.output_path) as f:
            chunks = (min(int(1e6), n_labels),)
            f.create_dataset(self.output_key, data=node_labels,
                             compression='gzip',  chunks=chunks)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_path, self.output_key))


class LiftedFeaturesFromNodeLabelsWorkflow(WorkflowBase):
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    nh_out_key = luigi.Parameter()
    feat_out_key = luigi.Parameter()
    prefix = luigi.Parameter()
    nh_graph_depth = luigi.IntParameter(default=4)
    ignore_label = luigi.IntParameter(default=0)
    label_ignore_label = luigi.Parameter(default=None)
    label_overlap_threshold = luigi.FloatParameter(default=None)
    clear_labels_path = luigi.Parameter(default=None)
    clear_labels_key = luigi.Parameter(default=None)
    mode = luigi.Parameter(default='all')

    def _clear_lifted_edges(self, dep):
        # 1.) get the node labels from overlapping labels from `clear_labels_path`
        # with the over-segmentation in `ws_path`
        # NOTE for now we don't allow for an ignore label in clear_labels
        labels_key = 'node_overlaps/clear_%s' % self.prefix
        dep = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                max_jobs=self.max_jobs, dependency=dep,
                                target=self.target,
                                ws_path=self.ws_path, ws_key=self.ws_key,
                                input_path=self.clear_labels_path,
                                input_key=self.clear_labels_key,
                                prefix='clear_%s' % self.prefix,
                                output_path=self.output_path,
                                output_key=labels_key, max_overlap=True,
                                ignore_label=None)
        clear_task = getattr(clear_tasks,
                             self._get_task_name('ClearLiftedEdgesFromLabels'))

        # 2.) clear lifted edges that connect nodes mapped to different ids
        # in the clear labels
        dep = clear_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                         max_jobs=self.max_jobs, dependency=dep,
                         node_labels_path=self.output_path, node_labels_key=labels_key,
                         lifted_edge_path=self.output_path, lifted_edge_key=self.nh_out_key)
        return dep

    def _node_labels(self, labels_key, dep):
        if self.label_overlap_threshold is None:
            labels_key1 = labels_key
            label_ignore_label = self.label_ignore_label
            max_overlap, serialize_counts = True, False
        else:
            labels_key1 = labels_key + '_tmp'
            label_ignore_label = None
            max_overlap, serialize_counts = False, True

        dep = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                max_jobs=self.max_jobs, dependency=dep,
                                target=self.target,
                                ws_path=self.ws_path, ws_key=self.ws_key,
                                input_path=self.labels_path, input_key=self.labels_key,
                                prefix=self.prefix, output_path=self.output_path,
                                output_key=labels_key1, max_overlap=max_overlap,
                                serialize_counts=serialize_counts, ignore_label=label_ignore_label)

        if self.label_overlap_threshold is not None:
            dep = NodeLabelsWithThreshold(input_path=self.output_path, input_key=labels_key1,
                                          output_path=self.output_path, output_key=labels_key,
                                          threshold=self.label_overlap_threshold,
                                          ignore_label=self.label_ignore_label,
                                          dependency=dep)

        return dep

    def requires(self):
        # 1.) get the node labels from overlapping labels from `labels_path`
        # with the over-segmentation in `ws_path`
        labels_key = 'node_overlaps/%s' % self.prefix
        dep = self._node_labels(labels_key, self.dependency)

        # 2.) find the sparse lifted neighborhood based on the node overlaps
        # and the neighborhood graph depth
        nh_task = getattr(nh_tasks,
                          self._get_task_name('SparseLiftedNeighborhood'))
        dep = nh_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                      max_jobs=self.max_jobs, dependency=dep,
                      graph_path=self.graph_path, graph_key=self.graph_key,
                      node_label_path=self.output_path, node_label_key=labels_key,
                      output_path=self.output_path, output_key=self.nh_out_key,
                      nh_graph_depth=self.nh_graph_depth, mode=self.mode,
                      prefix=self.prefix, node_ignore_label=self.ignore_label)

        # if we have clear labels, use them to filter the lifted edges
        if self.clear_labels_path is not None:
            assert self.clear_labels_key is not None
            dep = self._clear_lifted_edges(dep)

        # 3.) find the lifted features based on neighborhood and node labels
        cost_task = getattr(cost_tasks,
                            self._get_task_name('CostsFromNodeLabels'))
        dep = cost_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                        max_jobs=self.max_jobs, dependency=dep,
                        nh_path=self.output_path, nh_key=self.nh_out_key,
                        node_label_path=self.output_path, node_label_key=labels_key,
                        output_path=self.output_path, output_key=self.feat_out_key,
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
                        cost_tasks.CostsFromNodeLabelsLocal.default_task_config(),
                        'clear_lifted_edges_from_labels':
                        clear_tasks.ClearLiftedEdgesFromLabelsLocal.default_task_config()})
        return configs
