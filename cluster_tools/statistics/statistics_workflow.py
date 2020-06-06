import luigi

from .. cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from . import block_statistics as stat_tasks
from . import merge_statistics as merge_tasks


class DataStatisticsWorkflow(WorkflowBase):
    path = luigi.Parameter()
    key = luigi.Parameter()
    output_path = luigi.Parameter()

    def get_shape(self):
        with vu.file_reader(self.path, 'r') as f:
            shape = f[self.key].shape
        return shape

    def requires(self):
        stat_task = getattr(stat_tasks,
                            self._get_task_name('BlockStatistics'))
        dep = stat_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        path=self.path, key=self.key)

        merge_task = getattr(merge_tasks,
                             self._get_task_name('MergeStatistics'))
        dep = merge_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         output_path=self.output_path,
                         shape=self.get_shape(),
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(DataStatisticsWorkflow, DataStatisticsWorkflow).get_config()
        configs.update({'block_statistics':
                        stat_tasks.BlockStatisticsLocal.default_task_config(),
                        'merge_statistics':
                        merge_tasks.MergeStatisticsLocal.default_task_config()})
        return configs
