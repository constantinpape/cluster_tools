import luigi

from .. cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from . import block_morphology as block_tasks
from . import merge_morphology as merge_tasks
from . import correct_anchors as correct_tasks
from . import write_corrections as write_tasks


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


class CorrectAnchorsWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    morphology_path = luigi.Parameter()
    morphology_key = luigi.Parameter()
    output_path = luigi.Parameter(default='')
    output_key = luigi.Parameter(default='')

    def requires(self):
        dep = self.dependency
        correct_task = getattr(correct_tasks,
                               self._get_task_name('CorrectAnchors'))
        dep = correct_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                           max_jobs=self.max_jobs, dependency=dep,
                           input_path=self.input_path, input_key=self.input_key,
                           morphology_path=self.morphology_path, morphology_key=self.morphology_key)

        write_task = getattr(write_tasks,
                             self._get_task_name('WriteCorrections'))
        dep = write_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                         max_jobs=self.max_jobs, dependency=dep,
                         morphology_path=self.morphology_path, morphology_key=self.morphology_key,
                         output_path=self.output_path, output_key=self.output_key)
        return dep

    @staticmethod
    def get_config():
        configs = super(CorrectAnchorsWorkflow, CorrectAnchorsWorkflow).get_config()
        configs.update({'correct_anchors':
                        correct_tasks.CorrectAnchorsLocal.default_task_config(),
                        'write_corrections':
                        write_tasks.WriteCorrectionsLocal.default_task_config()})
        return configs
