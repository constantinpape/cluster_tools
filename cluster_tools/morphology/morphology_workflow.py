import os
import json
import luigi

from .. cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from . import block_morphology as block_tasks
from . import merge_morphology as merge_tasks


class MorphologyWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
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
                         output_key=tmp_key)
        merge_task = getattr(merge_tasks,
                             self._get_task_name('MergeMorphology'))
        with vu.file_reader(self.input_path) as f:
            number_of_labels = f[self.input_key].attrs['maxId'] + 1
        dep = merge_task(max_jobs=self.max_jobs,
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
