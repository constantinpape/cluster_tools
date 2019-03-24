import luigi

from ..cluster_tasks import WorkflowBase
from .. import copy_volume as copy_tasks
from . import insert_affinities as insert_tasks


class InsertAffinitiesWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    objects_path = luigi.Parameter()
    objects_key = luigi.Parameter()
    offsets = luigi.ListParameter(default=[[-1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, -1]])
    output_path = luigi.Parameter(default='')
    output_key = luigi.Parameter(default='')

    def requires(self):
        # check if we have an ouput path
        if self.output_path == '':  # no -> write to the input pace in-place
            affinity_path = self.input_path
            affinity_key = self.input_key
            dep = self.dependency
        else:  # yes -> first copy affinities and then write to them in-place
            assert self.input_key != ''
            copy_task = getattr(copy_tasks,
                                self._get_task_name('CopyVolume'))
            dep = copy_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                            config_dir=self.config_dir, dependency=self.dependency,
                            input_path=self.input_path, input_key=self.input_key,
                            output_path=self.output_path, output_key=self.output_key,
                            prefix='copy-affinities')
            affinity_path = self.output_path
            affinity_key = self.output_key

        insert_task = getattr(insert_tasks,
                              self._get_task_name('InsertAffinities'))
        dep = insert_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=dep,
                          affinity_path=affinity_path, affinity_key=affinity_key,
                          objects_path=self.objects_path, objects_key=self.objects_key,
                          offsets=self.offsets)
        return dep

    @staticmethod
    def get_config():
        configs = super(InsertAffinitiesWorkflow, InsertAffinitiesWorkflow).get_config()
        configs.update({'insert_affinities': insert_tasks.InsertAffinitiesLocal.default_task_config()})
        return configs
