import os
import luigi

from ..cluster_tasks import WorkflowBase
from . import block_tables as table_tasks
from . import eval_primitives as primitives_tasks


class EvaluationWorkflow(WorkflowBase):
    seg_path = luigi.Parameter()
    seg_key = luigi.Parameter()
    gt_path = luigi.Parameter()
    gt_key = luigi.Parameter()
    out_path = luigi.Parameter()

    def requires(self):

        tmp_path = os.path.join(self.tmp_folder, 'data.n5')
        tmp_key = 'block_tables'

        table_task = getattr(table_tasks,
                             self._get_task_name('BlockTables'))
        dep = table_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                         config_dir=self.config_dir, dependency=self.dependency,
                         seg_path=self.seg_path, seg_key=self.seg_key,
                         gt_path=self.gt_path, gt_key=self.gt_key,
                         output_path=tmp_path, output_key=tmp_key)

        primitives_task = getattr(primitives_tasks,
                                  self._get_task_name('EvalPrimitives'))
        dep = primitives_task(tmp_folder=self.tmp_folder, max_jobs=self.max_jobs,
                              config_dir=self.config_dir, dependency=dep,
                              input_path=tmp_path, input_key=tmp_key,
                              out_path=self.out_path)
        return dep

    @staticmethod
    def get_config():
        configs = super(EvaluationWorkflow, EvaluationWorkflow).get_config()
        configs.update({'block_tables': table_tasks.BlockTablesLocal.default_task_config(),
                        'eval_primitives': primitives_tasks.EvalPrimitivesLocal.default_task_config()})
        return configs
