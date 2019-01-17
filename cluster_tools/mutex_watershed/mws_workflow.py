import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from ..thresholded_components import merge_offsets as offset_tasks

from .import mws_blocks as block_tasks


class MwsWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    offsets = luigi.ListParameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    def requires(self):
        block_task = getattr(block_tasks, self._get_task_name('MwsBlocks'))
        dep = block_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                         config_dir=self.config_dir, dependency=self.dependency,
                         input_path=self.input_path, input_key=self.input_key,
                         output_path=self.output_path, output_key=self.output_key,
                         mask_path=self.mask_path, mask_key=self.mask_key,
                         offsets=self.offsets)

        # merge id-offsets
        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape[1:]
        offset_task = getattr(offset_tasks, self._get_task_name('MergeOffsets'))

        # temporary path for id-offsets
        offset_path = os.path.join(self.tmp_folder, 'mws_offsets.json')
        dep = offset_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=dep,
                          shape=shape, save_path=offset_path,
                          save_prefix='mws_offsets')
        return dep

    @staticmethod
    def get_config():
        configs = super(MwsWorkflow, MwsWorkflow).get_config()
        configs.update({'mws_blocks': block_tasks.MwsBlocksLocal.default_task_config()})
        return configs
