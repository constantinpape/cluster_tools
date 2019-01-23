import os
import json
import numpy as np
import luigi

from ..cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from ..utils import segmentation_utils as su
from ..thresholded_components import merge_offsets as offset_tasks
from ..thresholded_components import merge_assignments as merge_tasks
from .. import stitching as stitch_tasks
from .. import write as write_tasks

from .import mws_blocks as block_tasks
from .import mws_faces as face_tasks


class MwsWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    offsets = luigi.ListParameter()
    stitch_by_overlap = luigi.BoolParameter(default=True)
    overlap_threshold = luigi.FloatParameter(default=.9)
    halo = luigi.ListParameter(default=None)
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    def _stitch_by_overlap(self, dep, id_offset_path, halo):
        # merge block faces via max overlap
        shape = vu.get_shape(self.input_path, self.input_key)[1:]
        os.makedirs(os.path.join(self.tmp_folder, 'mws_overlaps'), exist_ok=True)
        ovlp_prefix = 'mws_overlaps/ovlp'
        stitch_task = getattr(stitch_tasks, self._get_task_name('StitchFaces'))
        dep = stitch_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=dep,
                          shape=shape, overlap_prefix=ovlp_prefix,
                          save_prefix='mws_assignments', offsets_path=id_offset_path,
                          overlap_threshold=self.overlap_threshold, halo=halo)
        return dep

    # FIXME this does not work properly yet
    def _stitch_by_mws(self, dep, id_offset_path):
        # merge block faces via mutex ws
        face_task = getattr(face_tasks, self._get_task_name('MwsFaces'))
        dep = face_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                        config_dir=self.config_dir, dependency=dep,
                        input_path=self.input_path, input_key=self.input_key,
                        output_path=self.output_path, output_key=self.output_key,
                        mask_path=self.mask_path, mask_key=self.mask_key,
                        offsets=self.offsets, id_offsets_path=id_offset_path)
        return dep

    def requires(self):
        # make sure we have affogato
        assert su.compute_mws_segmentation is not None, "Need affogato for mutex watershed"

        if self.halo is None:
            halo = np.max(np.abs(self.offsets), axis=0) + 1
            halo = halo.tolist()
        else:
            halo = self.halo

        block_task = getattr(block_tasks, self._get_task_name('MwsBlocks'))
        dep = block_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                         config_dir=self.config_dir, dependency=self.dependency,
                         input_path=self.input_path, input_key=self.input_key,
                         output_path=self.output_path, output_key=self.output_key,
                         mask_path=self.mask_path, mask_key=self.mask_key,
                         offsets=self.offsets, halo=halo,
                         serialize_overlap=self.stitch_by_overlap)

        # merge id-offsets
        with vu.file_reader(self.input_path, 'r') as f:
            shape = f[self.input_key].shape[1:]
        offset_task = getattr(offset_tasks, self._get_task_name('MergeOffsets'))

        # temporary path for id-offsets
        id_offset_path = os.path.join(self.tmp_folder, 'mws_offsets.json')
        dep = offset_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=dep,
                          shape=shape, save_path=id_offset_path,
                          save_prefix='mws_offsets')

        # get assignments either by re-running mws on the overlaps
        # or by stitching via biggest overlap
        dep = self._stitch_by_overlap(dep, id_offset_path, halo) if self.stitch_by_overlap else\
            self._stitch_by_mws(dep, id_offset_path)

        merge_task = getattr(merge_tasks,
                             self._get_task_name('MergeAssignments'))
        assignment_key = os.path.join(self.tmp_folder, 'assignments_mws')
        dep = merge_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                         max_jobs=self.max_jobs, dependency=dep,
                         output_path=self.output_path, output_key=assignment_key,
                         shape=shape, offset_path=id_offset_path,
                         save_prefix='mws_assignments')
        # return dep
        # we write in-place to the output dataset
        write_task = getattr(write_tasks,
                             self._get_task_name('Write'))
        dep = write_task(tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         max_jobs=self.max_jobs,
                         input_path=self.output_path, input_key=self.output_key,
                         output_path=self.output_path, output_key=self.output_key,
                         assignment_path=self.output_path, assignment_key=assignment_key,
                         identifier='mws', offset_path=id_offset_path,
                         dependency=dep)

        return dep

    @staticmethod
    def get_config():
        configs = super(MwsWorkflow, MwsWorkflow).get_config()
        # TODO add other configs
        configs.update({'mws_blocks': block_tasks.MwsBlocksLocal.default_task_config(),
                        'mws_faces': face_tasks.MwsFacesLocal.default_task_config()})
        return configs
