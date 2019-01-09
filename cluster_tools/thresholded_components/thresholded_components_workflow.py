import os
import json
import luigi

from ..utils import volume_utils as vu
from ..cluster_tasks import WorkflowBase
from .. import write as write_tasks
from . import block_components as block_tasks
from . import merge_offsets as offset_tasks
from . import block_faces as face_tasks
from . import merge_assignments as assignment_tasks


class ThresholdedComponentsWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    threshold = luigi.FloatParameter()
    threshold_mode = luigi.Parameter(default='greater')

    def requires(self):
        block_task = getattr(block_tasks,
                             self._get_task_name('BlockComponents'))
        offset_task = getattr(offset_tasks,
                              self._get_task_name('MergeOffsets'))
        face_task = getattr(face_tasks,
                            self._get_task_name('BlockFaces'))
        assignment_task = getattr(assignment_tasks,
                                  self._get_task_name('MergeAssignments'))
        write_task = getattr(write_tasks,
                             self._get_task_name('Write'))

        with vu.file_reader(self.input_path, 'r') as f:
            ds = f[self.input_key]
            shape = list(ds.shape)

        # temporary path for offsets
        offset_path = os.path.join(self.tmp_folder, 'cc_offsets.json')
        # path and key for assignments
        assignment_path = self.output_path
        assignment_key = 'cc_assignments'

        dep = block_task(tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         max_jobs=self.max_jobs,
                         input_path=self.input_path, input_key=self.input_key,
                         output_path=self.output_path, output_key=self.output_key,
                         threshold=self.threshold, threshold_mode=self.threshold_mode,
                         dependency=self.dependency)
        dep = offset_task(tmp_folder=self.tmp_folder,
                          config_dir=self.config_dir,
                          max_jobs=self.max_jobs,
                          shape=shape, save_path=offset_path,
                          dependency=dep)
        dep = face_task(tmp_folder=self.tmp_folder,
                        config_dir=self.config_dir,
                        max_jobs=self.max_jobs,
                        input_path=self.output_path, input_key=self.output_key,
                        offsets_path=offset_path, dependency=dep)
        dep = assignment_task(tmp_folder=self.tmp_folder,
                              config_dir=self.config_dir,
                              max_jobs=self.max_jobs,
                              output_path=assignment_path,
                              output_key=assignment_key,
                              shape=shape, offset_path=offset_path,
                              dependency=dep)
        # we write in-place to the output dataset
        dep = write_task(tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         max_jobs=self.max_jobs,
                         input_path=self.output_path, input_key=self.output_key,
                         output_path=self.output_path, output_key=self.output_key,
                         assignment_path=assignment_path, assignment_key=assignment_key,
                         identifier='connected_components', offset_path=offset_path,
                         dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(ThresholdedComponentsWorkflow, ThresholdedComponentsWorkflow).get_config()
        configs.update({'block_components':
                        block_tasks.BlockComponentsLocal.default_task_config(),
                        'merge_offsets':
                        offset_tasks.MergeOffsetsLocal.default_task_config(),
                        'block_faces':
                        face_tasks.BlockFacesLocal.default_task_config(),
                        'merge_assignments':
                        assignment_tasks.MergeAssignmentsLocal.default_task_config(),
                        'write':
                        write_tasks.WriteLocal.default_task_config()})
        return configs
