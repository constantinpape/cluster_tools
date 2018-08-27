import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import block_edge_features as feat_tasks
from . import merge_edge_features as merge_tasks


# TODO add option to skip ignore label in graph
# and implement in nifty
class EdgeFeaturesWorkflow(WorkflowBase):

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    max_jobs_merge = luigi.IntParameter()

    # for now we only support n5 / zarr input labels
    @staticmethod
    def _check_input(path):
        ending = path.split('.')[-1]
        assert ending.lower() in ('zr', 'zarr', 'n5'),\
            "Only support n5 and zarr files, not %s" % ending

    def requires(self):
        self._check_input(self.input_path)
        self._check_input(self.labels_path)

        feat_task = getattr(feat_tasks,
                            self._get_task_name('BlockEdgeFeatures'))
        t1 = feat_task(tmp_folder=self.tmp_folder,
                       max_jobs=self.max_jobs,
                       config_dir=self.config_dir,
                       input_path=self.input_path,
                       input_key=self.input_key,
                       labels_path=self.labels_path,
                       labels_key=self.labels_key,
                       graph_path=self.graph_path,
                       output_path=self.output_path,
                       dependency=self.dependency)
        merge_task = getattr(merge_tasks,
                     self._get_task_name('MergeEdgeFeatures'))
        t2 = merge_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs_merge,
                        config_dir=self.config_dir,
                        graph_path=self.graph_path,
                        graph_key=self.graph_key,
                        output_path=self.output_path,
                        output_key=self.output_key,
                        dependency=t1)
        return t2

    @staticmethod
    def get_config():
        configs = super(EdgeFeaturesWorkflow, EdgeFeaturesWorkflow).get_config()
        configs.update({'block_edge_features': feat_tasks.BlockEdgeFeaturesLocal.default_task_config(),
                       'merge_edge_features': merge_tasks.MergeEdgeFeaturesLocal.default_task_config()})
        return configs
