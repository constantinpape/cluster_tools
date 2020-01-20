import luigi

import cluster_tools.utils.volume_utils as vu

from ..cluster_tasks import WorkflowBase
from . import block_edge_features as feat_tasks
from . import merge_edge_features as merge_tasks
from . import region_features as reg_tasks
from . import merge_region_features as merge_reg_tasks


class EdgeFeaturesWorkflow(WorkflowBase):

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    max_jobs_merge = luigi.IntParameter(default=1)

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
        dep = feat_task(tmp_folder=self.tmp_folder,
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
        dep = merge_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs_merge,
                         config_dir=self.config_dir,
                         graph_path=self.graph_path,
                         graph_key=self.graph_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(EdgeFeaturesWorkflow, EdgeFeaturesWorkflow).get_config()
        configs.update({'block_edge_features': feat_tasks.BlockEdgeFeaturesLocal.default_task_config(),
                        'merge_edge_features': merge_tasks.MergeEdgeFeaturesLocal.default_task_config()})
        return configs


class RegionFeaturesWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    # it may be advisable to use less jobs for merging the features
    # because this is very I/O bound
    max_jobs_merge = luigi.IntParameter(default=None)
    channel = luigi.IntParameter(default=None)

    def read_number_of_labels(self):
        with vu.file_reader(self.labels_path, 'r') as f:
            n_labels = f[self.labels_key].attrs['maxId'] + 1
        return int(n_labels)

    def requires(self):
        feat_task = getattr(reg_tasks,
                            self._get_task_name('RegionFeatures'))
        dep = feat_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        input_path=self.input_path,
                        input_key=self.input_key,
                        labels_path=self.labels_path,
                        labels_key=self.labels_key,
                        channel=self.channel,
                        dependency=self.dependency)
        merge_task = getattr(merge_reg_tasks,
                             self._get_task_name('MergeRegionFeatures'))
        n_labels = self.read_number_of_labels()
        max_jobs_merge = self.max_jobs if self.max_jobs_merge is None else self.max_jobs_merge
        dep = merge_task(tmp_folder=self.tmp_folder,
                         max_jobs=max_jobs_merge,
                         config_dir=self.config_dir,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         number_of_labels=n_labels,
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(RegionFeaturesWorkflow, RegionFeaturesWorkflow).get_config()
        configs.update({'region_features': reg_tasks.RegionFeaturesLocal.default_task_config(),
                        'merge_region_features': merge_reg_tasks.MergeRegionFeaturesLocal.default_task_config()})
        return configs
