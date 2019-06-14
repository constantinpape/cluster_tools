import os
import luigi

from ..cluster_tasks import WorkflowBase
from ..utils import volume_utils as vu
from ..morphology import MorphologyWorkflow
from . import skeletonize as skeleton_tasks


class SkeletonWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    resolution = luigi.ListParameter()
    size_threshold = luigi.IntParameter(default=None)
    max_id = luigi.IntParameter(default=None)
    method = luigi.Parameter(default='thinning')

    def require_max_id(self):
        with vu.file_reader(self.input_path) as f:
            ds = f[self.input_key]
            attrs_max_id = ds.attrs.get('maxId', None)

            if self.max_id is None and attrs_max_id is None:
                raise RuntimeError("Input dataset does not have maxId attribute, so it needs to be passed externally")
            elif self.max_id is not None and attrs_max_id is None:
                ds.attrs['maxId'] = self.maxId
                max_id = self.maxId
            elif self.max_id is not None and attrs_max_id is not None:
                if self.max_id != attrs_max_id:
                    raise RuntimeError("MaxIds do not agree")
                max_id = self.max_id
            else:
                max_id = attrs_max_id
        return max_id

    def requires(self):

        # make sure the segmentation has max-id attribute
        max_id = self.require_max_id()
        n_labels = max_id + 1

        # compute the morphology
        tmp_path = os.path.join(self.tmp_folder, 'data.n5')
        tmp_key = 'morphology'
        dep = MorphologyWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                 max_jobs=self.max_jobs, target=self.target,
                                 input_path=self.input_path, input_key=self.input_key,
                                 output_path=tmp_path, output_key=tmp_key,
                                 dependency=self.dependency)

        skel_task = getattr(skeleton_tasks,
                            self._get_task_name('Skeletonize'))
        dep = skel_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                        config_dir=self.config_dir, dependency=dep,
                        input_path=self.input_path, input_key=self.input_key,
                        morphology_path=tmp_path, morphology_key=tmp_key,
                        output_path=self.output_path, output_key=self.output_key,
                        number_of_labels=n_labels, resolution=self.resolution,
                        size_threshold=self.size_threshold, method=self.method)
        return dep

    @staticmethod
    def get_config():
        configs = super(SkeletonWorkflow, SkeletonWorkflow).get_config()
        configs.update({'skeletonize': skeleton_tasks.SkeletonizeLocal.default_task_config(),
                        **MorphologyWorkflow.get_config()})
        return configs
