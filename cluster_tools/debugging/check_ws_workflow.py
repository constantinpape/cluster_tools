import os
import json
import luigi

from . import check_components as component_tasks
from ..cluster_tasks import WorkflowBase
from ..paintera import unique_block_labels as unique_tasks
from ..paintera import label_block_mapping as mapping_tasks
from ..utils import volume_utils as vu


# TODO fail if check does not pass
class CheckWsWorkflow(WorkflowBase):
    """ Check that watershed only has single connected
    component per label. Currently, does NOT work for
    two pass watershed.
    """
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    debug_path = luigi.Parameter()

    def requires(self):
        unique_task = getattr(unique_tasks, self._get_task_name('UniqueBlockLabels'))
        mapping_task = getattr(mapping_tasks, self._get_task_name('LabelBlockMapping'))
        component_task = getattr(component_tasks, self._get_task_name('CheckComponents'))

        with vu.file_reader(self.ws_path, 'r') as f:
            max_id = f[self.ws_key].attrs['maxId']
            chunks = list(f[self.ws_key].chunks)

        dep = unique_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          input_path=self.ws_path, output_path=self.debug_path,
                          input_key=self.ws_key, output_key='unique-labels',
                          dependency=self.dependency, prefix='debug_ws')
        dep = mapping_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                           config_dir=self.config_dir,
                           input_path=self.debug_path, output_path=self.debug_path,
                           input_key='unique-labels', output_key='label-block-mapping',
                           number_of_labels=max_id + 1, dependency=dep,
                           prefix='debug_ws')
        dep = component_task(input_path=self.debug_path, input_key='label-block-mapping',
                             output_path=self.debug_path, output_key='violating-fragment-ids',
                             number_of_labels=max_id + 1,
                             chunks=chunks, dependency=dep,
                             tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir)
        return dep

    @staticmethod
    def get_config():
        configs = super(CheckWsWorkflow, CheckWsWorkflow).get_config()
        configs.update({'unique_block_labels': unique_tasks.UniqueBlockLabelsLocal.default_task_config(),
                        'label_block_mapping': mapping_tasks.LabelBlockMappingLocal.default_task_config(),
                        'check_components': component_tasks.CheckComponentsLocal.default_task_config()})
        return configs
