import luigi

from ..cluster_tasks import WorkflowBase
from . import insert_affinities as insert_tasks


class InsertAffinitiesWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    objects_path = luigi.Parameter()
    objects_key = luigi.Parameter()
    offsets = luigi.ListParameter(default=[[-1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]])

    def requires(self):
        insert_task = getattr(insert_tasks,
                              self._get_task_name('InsertAffinities'))
        dep = insert_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=self.dependency,
                          input_path=self.input_path, input_key=self.input_key,
                          output_path=self.output_path, output_key=self.output_key,
                          objects_path=self.objects_path, objects_key=self.objects_key,
                          offsets=self.offsets)
        return dep

    @staticmethod
    def get_config():
        configs = super(InsertAffinitiesWorkflow, InsertAffinitiesWorkflow).get_config()
        configs.update({'insert_affinities': insert_tasks.InsertAffinitiesLocal.default_task_config()})
        return configs
