import os
import luigi

from .. cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from . import block_morphology as block_tasks
from . import merge_morphology as merge_tasks
from . import region_centers as center_tasks


class MorphologyWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    max_jobs_merge = luigi.Parameter(default=None)
    prefix = luigi.Parameter(default='')

    def requires(self):
        block_task = getattr(block_tasks,
                             self._get_task_name('BlockMorphology'))
        tmp_key = 'morphology_%s' % self.prefix
        dep = block_task(max_jobs=self.max_jobs,
                         tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         dependency=self.dependency,
                         input_path=self.input_path,
                         input_key=self.input_key,
                         output_path=self.output_path,
                         output_key=tmp_key,
                         prefix=self.prefix)
        merge_task = getattr(merge_tasks,
                             self._get_task_name('MergeMorphology'))
        with vu.file_reader(self.input_path) as f:
            number_of_labels = f[self.input_key].attrs['maxId'] + 1
        max_jobs_merge = self.max_jobs if self.max_jobs_merge is None else self.max_jobs_merge
        dep = merge_task(max_jobs=max_jobs_merge,
                         tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         input_path=self.output_path,
                         input_key=tmp_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         number_of_labels=number_of_labels,
                         prefix=self.prefix,
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(MorphologyWorkflow, MorphologyWorkflow).get_config()
        configs.update({'block_morphology':
                        block_tasks.BlockMorphologyLocal.default_task_config(),
                        'merge_morphology':
                        merge_tasks.MergeMorphologyLocal.default_task_config()})
        return configs


class RegionCentersWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    ignore_label = luigi.Parameter(default=None)
    resolution = luigi.ListParameter(default=[1, 1, 1])

    def requires(self):
        dep = self.dependency

        tmp_path = os.path.join(self.tmp_folder, 'region_centers.n5')
        tmp_key = 'morphology'

        dep = MorphologyWorkflow(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                                 config_dir=self.config_dir, target=self.target,
                                 input_path=self.input_path, input_key=self.input_key,
                                 output_path=tmp_path, output_key=tmp_key,
                                 dependency=dep, prefix='region-centers')
        center_task = getattr(center_tasks,
                              self._get_task_name('RegionCenters'))
        dep = center_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                          max_jobs=self.max_jobs, dependency=dep,
                          input_path=self.input_path, input_key=self.input_key,
                          morphology_path=tmp_path, morphology_key=tmp_key,
                          output_path=self.output_path, output_key=self.output_key,
                          ignore_label=self.ignore_label, resolution=self.resolution)
        return dep

    @staticmethod
    def get_config():
        configs = super(RegionCentersWorkflow, RegionCentersWorkflow).get_config()
        configs.update({'region_centers':
                        center_tasks.RegionCentersLocal.default_task_config(),
                        **MorphologyWorkflow.get_config()})
        return configs
