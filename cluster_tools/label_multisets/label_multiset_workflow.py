import os
import luigi

from ..cluster_tasks import WorkflowBase
from ..utils.volume_utils import file_reader
from . import create_multiset as create_tasks
from . import downscale_multiset as downscale_tasks


class LabelMultisetWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_prefix = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    restrict_sets = luigi.ListParameter()

    def _downscale(self, in_path, in_key, out_key,
                   scale_factor, effective_scale_factor,
                   restrict_set, dep):
        downscale_task = getattr(downscale_tasks,
                                 self._get_task_name('DownscaleMultiset'))
        return downscale_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                              max_jobs=self.max_jobs, dependency=dep,
                              input_path=in_path, input_key=in_key,
                              output_path=self.output_path, output_key=out_key,
                              scale_factor=scale_factor, restrict_set=restrict_set,
                              effective_scale_factor=effective_scale_factor)

    def requires(self):
        n_scales = len(self.scale_factors)
        assert len(self.restrict_sets) == n_scales

        # make label multiset for original scale
        out_key = os.path.join(self.output_prefix, 's0')
        create_task = getattr(create_tasks,
                              self._get_task_name('CreateMultiset'))
        dep = create_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                          max_jobs=self.max_jobs, dependency=self.dependency,
                          input_path=self.input_path, input_key=self.input_key,
                          output_path=self.output_path, output_key=out_key)
        in_key = out_key

        # make down-scaled multi-sets
        effective_scale_factor = [1, 1, 1]
        for level, scale_factor, restrict_set in zip(range(1, n_scales + 1),
                                                     self.scale_factors,
                                                     self.restrict_sets):
            effective_scale_factor = [eff * sc for eff, sc in zip(effective_scale_factor,
                                                                  scale_factor)]
            out_key = os.path.join(self.output_prefix, 's%i' % level)
            dep = self._downscale(self.output_path, in_key, out_key,
                                  scale_factor, effective_scale_factor,
                                  restrict_set, dep)
            in_key = out_key

        # write metadata
        self.write_metadata()
        return dep

    def write_metadata(self):
        with file_reader(self.input_path, 'r') as f:
            max_id = f[self.input_key].attrs['maxId']

        with file_reader(self.output_path) as f:
            g = f.require_group(self.output_prefix)
            attrs = g.attrs
            attrs['maxId'] = max_id
            attrs['isLabelMultiset'] = True
            attrs['multiScale'] = True

    @staticmethod
    def get_config():
        configs = super(LabelMultisetWorkflow, LabelMultisetWorkflow).get_config()
        configs.update({'create_multiset':
                        create_tasks.CreateMultisetLocal.default_task_config(),
                        'downscale_multiset':
                        downscale_tasks.DownscaleMultisetLocal.default_task_config()})
        return configs
