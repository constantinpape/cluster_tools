import luigi

from ..cluster_tasks import WorkflowBase
from . import affine as affine_tasks
from . import linear as linear_tasks
from . import transformix as transformix_tasks


class LinearTransformationWorkflow(WorkflowBase):
    """ Apply linear intensity transform.
    """

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    transformation = luigi.Parameter()
    output_path = luigi.Parameter(default='')
    output_key = luigi.Parameter(default='')
    mask_path = luigi.Parameter(default='')
    mask_key = luigi.Parameter(default='')

    def requires(self):

        # check if we apply the transformation in-place
        out_path = self.input_path if self.output_path == '' else self.output_path
        out_key = self.input_key if self.output_key == '' else self.output_key

        linear_task = getattr(linear_tasks,
                              self._get_task_name('Linear'))
        dep = linear_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=self.dependency,
                          input_path=self.input_path, input_key=self.input_key,
                          mask_path=self.mask_path, mask_key=self.mask_key,
                          output_path=out_path, output_key=out_key,
                          transformation=self.transformation)
        return dep

    @staticmethod
    def get_config():
        configs = super(LinearTransformationWorkflow, LinearTransformationWorkflow).get_config()
        configs.update({'linear': linear_tasks.LinearLocal.default_task_config()})
        return configs


class AffineTransformationWorkflow(WorkflowBase):
    """ Apply linear intensity transform.
    """

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    transformation = luigi.ListParameter()
    shape = luigi.ListParameter()
    order = luigi.IntParameter(default=0)

    def requires(self):
        affine_task = getattr(affine_tasks, self._get_task_name('Affine'))
        dep = affine_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                          config_dir=self.config_dir, dependency=self.dependency,
                          input_path=self.input_path, input_key=self.input_key,
                          output_path=self.output_path, output_key=self.output_key,
                          transformation=self.transformation, shape=self.shape,
                          order=self.order)
        return dep

    @staticmethod
    def get_config():
        configs = super(AffineTransformationWorkflow, AffineTransformationWorkflow).get_config()
        configs.update({'affine': affine_tasks.AffineLocal.default_task_config()})
        return configs


class TransformixTransformationWorkflow(WorkflowBase):
    """ Apply linear intensity transform.
    """

    input_path_file = luigi.Parameter()
    output_path_file = luigi.Parameter()
    transformation_file = luigi.Parameter()
    fiji_executable = luigi.Parameter()
    elastix_directory = luigi.Parameter()
    interpolation = luigi.Parameter(default='nearest')
    output_format = luigi.Parameter(default='bdv')
    shape = luigi.Parameter(default=None)
    resolution = luigi.Parameter(default=None)
    scale_factor = luigi.FloatParameter(default=1.e-3)

    formats = transformix_tasks.TransformixBase.formats
    result_types = transformix_tasks.TransformixBase.result_types
    interpolation_modes = transformix_tasks.TransformixBase.interpolation_modes

    def requires(self):
        transformix_task = getattr(transformix_tasks, self._get_task_name('Transformix'))

        if self.resolution is None:
            resolution = None
        else:
            resolution = [res * self.scale_factor for res in self.resolution]

        dep = transformix_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                               config_dir=self.config_dir, dependency=self.dependency,
                               input_path_file=self.input_path_file, output_path_file=self.output_path_file,
                               transformation_file=self.transformation_file, fiji_executable=self.fiji_executable,
                               elastix_directory=self.elastix_directory, interpolation=self.interpolation,
                               output_format=self.output_format, shape=self.shape,
                               resolution=resolution)
        return dep

    @staticmethod
    def get_config():
        configs = super(TransformixTransformationWorkflow, TransformixTransformationWorkflow).get_config()
        configs.update({'transformix': transformix_tasks.TransformixLocal.default_task_config()})
        return configs
