import os
import luigi
import numpy as np

from ..cluster_tasks import WorkflowBase
from ..node_labels import NodeLabelWorkflow
from . import measures as measure_tasks
from . import object_vi as vi_tasks
from . import object_iou as iou_tasks


class ObjectViWorkflow(WorkflowBase):
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

        # 2.) compute the object vis from overlaps
        vi_task = getattr(vi_tasks, self._get_task_name('ObjectVi'))
        dep = vi_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                      max_jobs=self.max_jobs, dependency=dep,
                      input_path=tmp_path, overlap_key=ovlp_key,
                      output_path=self.output_path)
        return dep

    @staticmethod
    def get_config():
        configs = super(EvaluationWorkflow, EvaluationWorkflow).get_config()
        configs.update({'object_vi': vi_tasks.ObjectViLocal.default_task_config(),
                        **NodeLabelWorkflow.get_config()})
        return configs


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
        measure_task = getattr(measure_tasks,
                               self._get_task_name('Measures'))
        dep = measure_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                           max_jobs=self.max_jobs, dependency=dep,
                           input_path=tmp_path, overlap_key=ovlp_key,
                           output_path=self.output_path)

        return dep

    @staticmethod
    def get_config():
        configs = super(EvaluationWorkflow, EvaluationWorkflow).get_config()
        configs.update({'measures': measure_tasks.MeasuresLocal.default_task_config(),
                        **NodeLabelWorkflow.get_config()})
        return configs


class ObjectIouWorkflow(WorkflowBase):
    seg_path = luigi.Parameter()
    seg_key = luigi.Parameter()
    gt_path = luigi.Parameter()
    gt_key = luigi.Parameter()
    output_path = luigi.Parameter()

    def requires(self):
        # 1.) compute the full overlaps between the segmentation and the groundtruth
        tmp_path = os.path.join(self.tmp_folder, 'data.n5')
        ovlp_key = 'overlaps'
        dep = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                max_jobs=self.max_jobs, target=self.target,
                                ws_path=self.seg_path, ws_key=self.seg_key,
                                input_path=self.gt_path, input_key=self.gt_key,
                                output_path=tmp_path, output_key=ovlp_key,
                                max_overlap=False, ignore_label=None,
                                serialize_counts=True, dependency=self.dependency)

        # 2.) compute the object ious from overlaps
        iou_task = getattr(iou_tasks, self._get_task_name('ObjectIou'))
        dep = iou_task(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                       max_jobs=self.max_jobs, dependency=dep,
                       input_path=tmp_path, overlap_key=ovlp_key,
                       output_path=self.output_path)

        return dep

    @staticmethod
    def get_config():
        configs = super(ObjectIouWorkflow, ObjectIouWorkflow).get_config()
        configs.update({'object_iou': iou_tasks.ObjectIouLocal.default_task_config(),
                        **NodeLabelWorkflow.get_config()})
        return configs


# TODO
class MeanAPTask(luigi.Task):
    tmp_folder = luigi.Parameter()
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        raise NotImplementedError
        return self.dependency

    def run(self):
        pass

    def output(self):
        pass


class MeanAPWorkflow(WorkflowBase):
    seg_path = luigi.Parameter()
    seg_key = luigi.Parameter()
    gt_path = luigi.Parameter()
    gt_key = luigi.Parameter()
    output_path = luigi.Parameter()
    ap_thresholds = luigi.ListParameter(default=np.arange(0.5, 1., 0.05).tolist())

    def requires(self):
        tmp_out = os.path.join(self.tmp_folder, 'object_ious.json')
        dep = ObjectIouWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                max_jobs=self.max_jobs, target=self.target,
                                seg_path=self.seg_path, seg_key=self.seg_key,
                                gt_path=self.gt_path, gt_key=self.gt_key,
                                output_path=tmp_out, dependency=self.dependency)

        dep = MeanAPTask(tmp_folder=self.tmp_folder, input_path=tmp_out,
                         output_path=self.output_path, dependency=dep)

        return dep

    @staticmethod
    def get_config():
        configs = super(MeanAPWorkflow, MeanAPWorkflow).get_config()
        configs.update({**ObjectIouWorkflow.get_config()})
        return configs
