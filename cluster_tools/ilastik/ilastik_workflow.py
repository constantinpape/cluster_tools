import os
import json
import luigi

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
    # TODO we could also read this from the temporary
    # results, but I am too lazy right now
    n_channels = luigi.IntParameter()

    def requires(self):

        predict_task = getattr(predict_tasks,
                               self._get_task_name('Prediction'))
        output_prefix = os.path.splitext(self.output_path)[0]
        dep = predict_task(tmp_folder=self.tmp_folder,
                           max_jobs=self.max_jobs,
                           config_dir=self.config_dir,
                           input_path=self.input_path,
                           input_key=self.input_key,
                           output_prefix=output_prefix,
                           ilastik_folder=self.ilastik_folder,
                           ilastik_project=self.ilastik_project,
                           halo=self.halo)
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
