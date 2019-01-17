import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
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
        # TODO merge tasks
        return dep

    @staticmethod
    def get_config():
        configs = super(MwsWorkflowWorkflow, MwsWorkflowWorkflow).get_config()
        configs.update({'mws_blocks': block_tasks.MwsBlocksLocal.default_task_config()})
        return configs
