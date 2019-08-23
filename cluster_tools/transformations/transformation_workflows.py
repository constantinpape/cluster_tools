import luigi

from ..cluster_tasks import WorkflowBase
from . import linear as linear_tasks


class LinearTransformationWorkflow(WorkflowBase):
    """ Apply linear intensity transform.
    """

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    transformation = luigi.Parameter()
    output_path = luigi.Parameter(default='')
    output_key = luigi.Parameter(default='')

    def requires(self):

        # check if we apply the transformation in-place
        out_path = self.input_path if self.output_path == '' else self.output_path
        out_key = self.input_key if self.output_key == '' else self.output_key

        linear_task = getattr(linear_tasks,
                              self._get_task_name('Linear'))
        dep = linear_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=self.dependency,
                          input_path=self.input_path, input_key=self.input_key,
                          output_path=out_path, output_key=out_key,
                          transformation=self.transformation)
        return dep

    @staticmethod
    def get_config():
        configs = super(LinearTransformationWorkflow, LinearTransformationWorkflow).get_config()
        configs.update({'linear': linear_tasks.LinearLocal.default_task_config()})
        return configs
