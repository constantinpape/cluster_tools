import os
import luigi

from ..cluster_tasks import WorkflowBase
from ..node_labels import NodeLabelWorkflow
from . import measures as measure_tasks


class EvaluationWorkflow(WorkflowBase):
    seg_path = luigi.Parameter()
    seg_key = luigi.Parameter()
    gt_path = luigi.Parameter()
    gt_key = luigi.Parameter()
    output_path = luigi.Parameter()
    ignore_label = luigi.BoolParameter(default=True)

    def requires(self):

        # 1.) compute the full overlaps between the segmentation and the groundtruth
        tmp_path = os.path.join(self.tmp_folder, 'data.n5')
        ovlp_key = 'overlaps'
        ignore_label = 0 if self.ignore_label else None
        dep = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                max_jobs=self.max_jobs, target=self.target,
                                ws_path=self.seg_path, ws_key=self.seg_key,
                                input_path=self.gt_path, input_key=self.gt_key,
                                output_path=tmp_path, output_key=ovlp_key,
                                max_overlap=False, ignore_label=ignore_label,
                                serialize_counts=True, dependency=self.dependency)

        # 2.) build contingency table from overlaps and compute validation measures
        meausre_task = getattr(measure_tasks,
                               self._get_task_name('Measures'))
        dep = meausre_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                           max_jobs=self.max_jobs, dependency=dep,
                           input_path=tmp_path, overlap_key=ovlp_key,
                           output_path=self.output_path)

        return dep

    @staticmethod
    def get_config():
        configs = super(EvaluationWorkflow, EvaluationWorkflow).get_config()
        configs.update({**NodeLabelWorkflow.get_config()})
        return configs
