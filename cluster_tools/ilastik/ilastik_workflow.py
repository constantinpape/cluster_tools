import os
import json
import luigi

import cluster_tools.utils.volume_utils as vu
from ..cluster_tasks import WorkflowBase
from . import prediction as predict_tasks
from . import merge_predictions as merge_tasks


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
