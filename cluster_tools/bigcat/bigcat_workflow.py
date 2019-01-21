import os
import json

import numpy as np
import luigi
import h5py

from ..utils import volume_utils as vu
from ..cluster_tasks import WorkflowBase
from .. import copy_volume as copy_tasks


class BigcatLabelAssignment(luigi.Task):
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    def requires(self):
        return self.dependency

    def run(self):
        with vu.file_reader(self.input_path, 'r') as f:
            assignments = f[self.input_key][:]
        assert assignments.ndim == 1

        node_ids = np.arange(len(assignments), dtype='uint64')
        lut = np.zeros((2, len(assignments)), dtype='uint64')
        lut[0, :] = node_ids
        max_node_id = len(assignments)
        lut[1, :] = assignments + max_node_id
        next_id = int(lut.max()) + 1

        with h5py.File(self.output_path) as f:
            f.attrs['next_id'] = next_id
            ds = f.require_dataset('fragment_segment_lut', shape=lut.shape,
                                   compression='gzip', maxshape=(2, None),
                                   dtype='uint64')
            ds[:] = lut

        with open(self.output().path, 'w') as f:
            print("Processed")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'bigcat_label_assignment.log'))


class BigcatMetadata(luigi.Task):
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    input_path = luigi.Parameter()
    raw_key = luigi.Parameter()
    seg_key = luigi.Parameter()
    resolution = luigi.ListParameter()
    offset = luigi.ListParameter(default=None)

    def requires(self):
        return self.dependency

    def run(self):
        offset = 3 * [0] if self.offset is None else self.offset
        assert len(self.resolution) == len(offset) == 3

        with h5py.File(self.input_path) as f:
            ds = f[self.raw_key]
            ds.attrs['resolution'] = self.resolution
            ds.attrs['offset'] = 3 * [0]

            ds = f[self.seg_key]
            ds.attrs['resolution'] = self.resolution
            ds.attrs['offset'] = offset

        with open(self.output().path, 'w') as f:
            print("Processed")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'bigcat_metadata.log'))


# conversion to bigcat:
# - need oversegmentation + assignments
class BigcatWorkflow(WorkflowBase):
    raw_path = luigi.Parameter()
    raw_key = luigi.Parameter()
    seg_path = luigi.Parameter()
    seg_key = luigi.Parameter()
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    output_path = luigi.Parameter()
    resolution = luigi.ListParameter()
    offset = luigi.ListParameter(default=None)

    def requires(self):
        copy_task = getattr(copy_tasks, self._get_task_name('CopyVolume'))
        # copy raw volume to output_path
        raw_out_key = 'volumes/raw'
        dep = copy_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                        config_dir=self.config_dir, dependency=self.dependency,
                        input_path=self.raw_path, input_key=self.raw_key,
                        output_path=self.output_path, output_key=raw_out_key,
                        prefix='bigcat_raw', dtype='uint8')

        # copy segmentation to output path
        seg_out_key = 'volumes/labels/fragments'
        dep = copy_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                        config_dir=self.config_dir, dependency=dep,
                        input_path=self.seg_path, input_key=self.seg_key,
                        output_path=self.output_path, output_key=seg_out_key,
                        prefix='bigcat_seg', dtype='uint64')

        # make fragment -> segment assignments
        label_out_key = 'fragment_segment_lut'
        dep = BigcatLabelAssignment(tmp_folder=self.tmp_folder, dependency=dep,
                                    input_path=self.assignment_path, input_key=self.assignment_key,
                                    output_path=self.output_path, output_key=label_out_key)

        # write additional meta data
        dep = BigcatMetadata(tmp_folder=self.tmp_folder, dependency=dep,
                             input_path=self.output_path, raw_key=raw_out_key,
                             seg_key=seg_out_key, resolution=self.resolution,
                             offset=self.offset)
        return dep

    @staticmethod
    def get_config():
        configs = super(BigcatWorkflow, BigcatWorkflow).get_config()
        configs.update({'copy_volume': copy_tasks.CopyVolumeLocal.default_task_config()})
        return configs
