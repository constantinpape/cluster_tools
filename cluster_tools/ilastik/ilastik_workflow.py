import os
import luigi
import uuid

import cluster_tools.utils.volume_utils as vu
from ..graph import GraphWorkflow
from ..cluster_tasks import WorkflowBase
from ..features import EdgeFeaturesWorkflow
from .. import copy_volume as copy_tasks

from . import prediction as predict_tasks
from . import merge_predictions as merge_tasks
from .carving import WriteCarving


class IlastikPredictionWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()

    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    ilastik_folder = luigi.Parameter()
    ilastik_project = luigi.Parameter()
    halo = luigi.ListParameter()
    n_channels = luigi.IntParameter()

    def requires(self):
        is_h5 = vu.is_h5(self.output_path)
        out_key = None if is_h5 else self.output_key
        predict_task = getattr(predict_tasks,
                               self._get_task_name('Prediction'))
        dep = predict_task(tmp_folder=self.tmp_folder,
                           max_jobs=self.max_jobs,
                           config_dir=self.config_dir,
                           input_path=self.input_path,
                           input_key=self.input_key,
                           output_path=self.output_path,
                           output_key=out_key,
                           ilastik_folder=self.ilastik_folder,
                           ilastik_project=self.ilastik_project,
                           halo=self.halo, n_channels=self.n_channels)
        # we only need to merge the predictions seperately if the
        # output file is hdf5
        if is_h5:
            output_prefix = os.path.splitext(self.output_path)[0]
            merge_task = getattr(merge_tasks,
                                 self._get_task_name('MergePredictions'))
            dep = merge_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             dependency=dep,
                             input_path=self.input_path,
                             input_key=self.input_key,
                             tmp_prefix=output_prefix,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             halo=self.halo,
                             n_channels=self.n_channels)

        return dep

    @staticmethod
    def get_config():
        configs = super(IlastikPredictionWorkflow, IlastikPredictionWorkflow).get_config()
        configs.update({'prediction':
                        predict_tasks.PredictionLocal.default_task_config(),
                        'merge_predictions':
                        merge_tasks.MergePredictionsLocal.default_task_config()})
        return configs


class IlastikCarvingWorkflow(WorkflowBase):
    """ Make carving project with watershed and graph
    """
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    watershed_path = luigi.Parameter()
    watershed_key = luigi.Parameter()
    output_path = luigi.Parameter()
    copy_inputs = luigi.BoolParameter(default=False)

    def requires(self):
        tmp_path = os.path.join(self.tmp_folder, 'exp_data.n5')
        graph_key = 'graph'
        feat_key = 'feats'
        # TODO make param ?
        max_jobs_merge = 1
        dep = GraphWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                            max_jobs=self.max_jobs, target=self.target, dependency=self.dependency,
                            input_path=self.watershed_path, input_key=self.watershed_key,
                            graph_path=tmp_path, output_key=graph_key)
        dep = EdgeFeaturesWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                   max_jobs=self.max_jobs, target=self.target, dependency=dep,
                                   input_path=self.input_path, input_key=self.input_key,
                                   labels_path=self.watershed_path,
                                   labels_key=self.watershed_key,
                                   graph_path=tmp_path, graph_key=graph_key,
                                   output_path=tmp_path, output_key=feat_key,
                                   max_jobs_merge=max_jobs_merge)

        # write the carving graph data and metadata
        uid = str(uuid.uuid1())
        dep = WriteCarving(input_path=tmp_path, graph_key=graph_key, features_key=feat_key,
                           raw_path=self.input_path, raw_key=self.input_key, uid=uid,
                           output_path=self.output_path, copy_inputs=self.copy_inputs, dependency=dep)

        copy_task = getattr(copy_tasks, self._get_task_name('CopyVolume'))
        # copy the watershed segmentation to ilastik file
        ilastik_seg_key = 'preprocessing/graph/labels'
        ilastik_seg_dtype = 'uint32'  # TODO is uint32 correct ?
        dep = copy_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                        max_jobs=1, dependency=dep,
                        input_path=self.watershed_path, input_key=self.watershed_key,
                        output_path=self.output_path, output_key=ilastik_seg_key,
                        dtype=ilastik_seg_dtype, prefix='watershed')

        # copy the input map to ilastik file
        if self.copy_inputs:
            ilastik_inp_key = 'Input Data/local_data/%s' % uid
            ilastik_inp_dtype = 'float32'  # is float32 correct ?
            dep = copy_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                            max_jobs=1, dependency=dep,
                            input_path=self.input_path, input_key=self.input_key,
                            output_path=self.output_path, output_key=ilastik_inp_key,
                            dtype=ilastik_inp_dtype, prefix='inputs')
        return dep

    @staticmethod
    def get_config():
        configs = super(IlastikCarvingWorkflow, IlastikCarvingWorkflow).get_config()
        configs.update({"copy_volume": copy_tasks.CopyVolumeLocal.default_task_config(),
                        **EdgeFeaturesWorkflow.get_config(),
                        **GraphWorkflow.get_config()})
        return configs
