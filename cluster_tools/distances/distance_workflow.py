import os
import json
import luigi

from . import object_distances as distance_tasks
from ..cluster_tasks import WorkflowBase


class MergePairwiseDistances(luigi.Task):
    tmp_folder = luigi.Parameter()
    max_jobs = luigi.IntParameter()
    output_path = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run(self):
        res_dict = {}
        for job_id in range(self.max_jobs):
            path = os.path.join(self.tmp_folder, 'object_distances_%i.json' % job_id)
            with open(path) as f:
                distances = json.load(f)
                res_dict.update(distances)
            with open(self.output_path, 'w') as f:
                json.dimp(res_dict, f)

    def output(self):
        return luigi.LocalTarget(self.output_path)


class PairwiseDistanceWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    morphology_path = luigi.Parameter()
    morphology_key = luigi.Parameter()
    output_path = luigi.Parameter()
    max_distance = luigi.FloatParameter()
    resolution = luigi.ListParamete()
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    def requires(self):
        distance_task = getattr(distance_tasks,
                                self._get_task_name('ObjectDistances'))
        dep = distance_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            input_path=self.input_path, input_key=self.input_key,
                            morphology_path=self.morphology_path, morphology_key=self.morphology_key,
                            max_distance=self.max_distance, resolution=self.resolution,
                            mask_path=self.mask_path, mask_key=self.mask_key)
        dep = MergePairwiseDistances(tmp_folder=self.tmp_folder,
                                     output_path=self.output_path,
                                     dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(PairwiseDistanceWorkflow, PairwiseDistanceWorkflow).get_config()
        configs.update({'object_distances': distance_tasks.ObjectDistancesLocal.get_default_config()})
        return configs
