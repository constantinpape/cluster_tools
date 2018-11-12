import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from ..downscaling import DownscalingWorkflow
# TODO
# from ..label_multisets import

from . import unique_block_labels as unique_tasks


class ConversionWorkflow(WorkflowBase):
    path = luigi.Parameter()
    label_in_key = luigi.Parameter()
    label_out_key = luigi.Parameter()
    assignment_key = luigi.Parameter()
    use_label_multiset = luigi.BoolParameter(default=False)
    # TODO do we handle raw data here ?
    # TODO upscaling labels

    def _copy_labels(self):
        pass

    def _(self, dependency):
        pass

    def requires(self):
        # first, we copy the labels to label-out-key,
        # (as label-multi-set if specified)
        t1 = self._copy_labels()
        # next, downscale the labels
        t2 = ''
        # next, compute the mapping of unique labels to blocks
        t3 = ''
        # next, compute the inverse mapping
        t4 = ''
        # finally, compute the fragment-segment-assignment
        t5 = ''
        return t5

    @staticmethod
    def get_config():
        configs = super(ConversionWorkflow, ConversionWorkflow).get_config()
        configs.update({'unique_block_labels': unique_tasks.default_task_config(),
                        **DownscalingWorkflow.get_config()})
        return configs
