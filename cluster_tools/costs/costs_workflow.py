import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import predict as predict_tasks
from . import probs_to_costs as transform_tasks


# TODO add option to skip ignore label in graph
# and implement in nifty
class EdgeCostsWorkflow(WorkflowBase):

    features_path = luigi.Parameter()
    features_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    rf_path = luigi.Parameter(default='')

    def _costs_with_rf(self):
        predict_task = getattr(predict_tasks,
                               self._get_task_name('Predict'))
        t1 = predict_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          rf_path=self.rf_path,
                          features_path=self.features_path,
                          features_key=self.features_key,
                          output_path=self.output_path,
                          output_key=self.output_key,
                          dependency=self.dependency)
        transform_task = getattr(transform_tasks,
                                 self._get_task_name('ProbsToCosts'))
        t2 = transform_task(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            input_path=self.output_path,
                            input_key=self.output_key,
                            features_path=self.features_path,
                            features_key=self.features_key,
                            output_path=self.output_path,
                            output_key=self.output_key,
                            dependency=t1)
        return t2

    def _costs(self):
        transform_task = getattr(transform_tasks,
                                 self._get_task_name('ProbsToCosts'))
        t1 = transform_task(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            input_path=self.features_path,
                            input_key=self.features_key,
                            features_path=self.features_path,
                            features_key=self.features_key,
                            output_path=self.output_path,
                            output_key=self.output_key,
                            dependency=self.dependency)
        return t1

    def requires(self):
        if self.rf_path == '':
            return self._costs()
        else:
            return self._costs_with_rf()

    def get_config(self):
        configs = super().get_config()
        configs.update({'probs_to_costs': (os.path.join(self.config_dir, 'probs_to_costs.config'),
                                           transform_tasks.ProbsToCostsLocal.default_task_config())})
        return configs
