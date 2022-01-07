import luigi

from ..utils import volume_utils as vu
from ..cluster_tasks import WorkflowBase
from ..watershed import watershed_from_seeds as ws_tasks

from . import connected_component_blocks as block_tasks
from . import merge_faces as face_tasks
from . import merge_assignments as assignment_tasks
from .. import write as write_tasks


class ConnectedComponentsWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    assignment_key = luigi.Parameter()
    threshold = luigi.FloatParameter(default=None)
    threshold_mode = luigi.Parameter(default=None)
    mask_path = luigi.Parameter(default="")
    mask_key = luigi.Parameter(default="")
    channel = luigi.Parameter(default=None)

    def requires(self):
        block_task = getattr(block_tasks, self._get_task_name("ConnectedComponentBlocks"))
        face_task = getattr(face_tasks, self._get_task_name("MergeFaces"))
        assignment_task = getattr(assignment_tasks, self._get_task_name("MergeAssignments"))
        write_task = getattr(write_tasks, self._get_task_name("Write"))

        with vu.file_reader(self.input_path, "r") as f:
            ds = f[self.input_key]
            shape = list(ds.shape)

        if self.channel is None:
            assert len(shape) == 3
        else:
            assert len(shape) == 4
            shape = shape[1:]

        dep = block_task(tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         max_jobs=self.max_jobs,
                         input_path=self.input_path, input_key=self.input_key,
                         output_path=self.output_path, output_key=self.output_key,
                         threshold=self.threshold, threshold_mode=self.threshold_mode,
                         mask_path=self.mask_path, mask_key=self.mask_key,
                         channel=self.channel,
                         dependency=self.dependency)
        dep = face_task(tmp_folder=self.tmp_folder,
                        config_dir=self.config_dir,
                        max_jobs=self.max_jobs,
                        input_path=self.output_path, input_key=self.output_key, dependency=dep)
        dep = assignment_task(tmp_folder=self.tmp_folder,
                              config_dir=self.config_dir,
                              max_jobs=self.max_jobs,
                              output_path=self.output_path,
                              output_key=self.assignment_key,
                              shape=shape, dependency=dep)
        # we write in-place to the output dataset
        dep = write_task(tmp_folder=self.tmp_folder,
                         config_dir=self.config_dir,
                         max_jobs=self.max_jobs,
                         input_path=self.output_path, input_key=self.output_key,
                         output_path=self.output_path, output_key=self.output_key,
                         assignment_path=self.output_path, assignment_key=self.assignment_key,
                         identifier="connected_components", dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(ConnectedComponentsWorkflow, ConnectedComponentsWorkflow).get_config()
        configs.update({"connected_components_blocks":
                        block_tasks.BlockComponentsLocal.default_task_config(),
                        "merge_faces":
                        face_tasks.BlockFacesLocal.default_task_config(),
                        "component_assignments":
                        assignment_tasks.MergeAssignmentsLocal.default_task_config(),
                        "write":
                        write_tasks.WriteLocal.default_task_config()})
        return configs


class ConnectedComponentsAndWatershedWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    assignment_key = luigi.Parameter()
    threshold = luigi.FloatParameter(default=None)
    threshold_mode = luigi.Parameter(default=None)
    mask_path = luigi.Parameter(default="")
    mask_key = luigi.Parameter(default="")
    channel = luigi.IntParameter(default=None)

    def requires(self):
        dep = ConnectedComponentsWorkflow(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                                          config_dir=self.config_dir, target=self.target,
                                          input_path=self.input_path, input_key=self.input_key,
                                          output_path=self.output_path, output_key=self.output_key,
                                          assignment_key=self.assignment_key, threshold=self.threshold,
                                          threshold_mode=self.threshold_mode, mask_path=self.mask_path,
                                          mask_key=self.mask_key, channel=self.channel,
                                          dependency=self.dependency)
        ws_task = getattr(ws_tasks,
                          self._get_task_name("WatershedFromSeeds"))
        dep = ws_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                      config_dir=self.config_dir, dependency=dep,
                      input_path=self.input_path, input_key=self.input_key,
                      seeds_path=self.output_path, seeds_key=self.output_key,
                      output_path=self.output_path, output_key=self.output_key,
                      mask_path=self.mask_path, mask_key=self.mask_key)
        return dep

    @staticmethod
    def get_config():
        configs = super(ConnectedComponentsAndWatershedWorkflow, ConnectedComponentsAndWatershedWorkflow).get_config()
        configs.update({"watershed_from_seeds":
                        ws_tasks.WatershedFromSeedsLocal.default_task_config(),
                        **ConnectedComponentsWorkflow.get_config()})
        return configs
