import os
import luigi

from ..cluster_tasks import WorkflowBase
from ..utils.volume_utils import file_reader
from . import label_multiset as multiset_tasks


class LabelMultisetWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_prefix = luigi.Parameter()
    scale_factors = luigi.ListParameter()
    restrict_sets = luigi.ListParameter()

    def _label_multiset(self, in_path, in_key, out_key,
                        scale_factor, restrict_set, dep):
        multiset_task = getattr(multiset_tasks,
                                self._get_task_name('LabelMultiset'))
        return multiset_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                             max_jobs=self.max_jobs, dependency=dep,
                             input_path=in_path, input_key=in_key,
                             output_path=self.output_path, output_key=self.output_key,
                             scale_factor=scale_factor, restrict_set=restrict_set)

    def requires(self):
        n_scales = len(scale_factors)
        assert len(restrict_sets) == n_scales

        # make label multiset for original scale
        out_key = os.path.join(output_prefix, 's0')
        dep = self._label_multiset(self.input_path, self.input_key,
                                   out_key, 1, None, self.dependency)
        in_key = out_key

        # make down-scaled multi-sets
        for level, scale_factor, restrict_set in zip(range(1, n_scales + 1),
                                                     scale_factors,
                                                     restrict_sets):
            out_key = os.path.join(output_prefix, 's%i' % level)
            dep = self._label_multiset(self.output_path, in_key, out_key,
                                       scale_factor, restrict_set, dep)
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
        configs.update({'label_multiset':
                        multiset_tasks.LabelMultisetLocal.default_task_config()})
        return configs
