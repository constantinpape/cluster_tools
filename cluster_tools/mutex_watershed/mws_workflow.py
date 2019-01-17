import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from .import mws_blocks as block_tasks


class MwsWorkflowWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    def requires(self):
        pass
        # return dep

    @staticmethod
    def get_config():
        configs = super(MwsWorkflowWorkflow, MwsWorkflowWorkflow).get_config()
        # configs.update({})
        return configs
