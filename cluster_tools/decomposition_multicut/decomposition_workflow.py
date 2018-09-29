import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import decompose as decompose_tasks
from . import solve_subproblems as subproblem_tasks
from . import insert as insert_tasks


class DecompositionWorkflow(WorkflowBase):
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    costs_path = luigi.Parameter()
    costs_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    decomposition_path = luigi.Parameter()

    def requires(self):

        decompose_task = getattr(decompose_tasks,
                                 self._get_task_name('Decompose'))
        subproblem_task = getattr(subproblem_tasks,
                                  self._get_task_name('SolveSubproblems'))
        insert_task = getattr(insert_tasks,
                              self._get_task_name('Insert'))

        t_0 = decompose_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             graph_path=self.graph_path,
                             graph_key=self.graph_key,
                             costs_path=self.costs_path,
                             costs_key=self.costs_key,
                             output_path=self.decomposition_path,
                             dependency=self.dependency)
        t_1 = subproblem_task(tmp_folder=self.tmp_folder,
                              max_jobs=self.max_jobs,
                              config_dir=self.config_dir,
                              graph_path=self.graph_path,
                              graph_key=self.graph_key,
                              costs_path=self.costs_path,
                              costs_key=self.costs_key,
                              decomposition_path=self.decomposition_path,
                              dependency=t_0)
        t_2 = insert_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          graph_path=self.graph_path,
                          graph_key=self.graph_key,
                          output_path=self.output_path,
                          output_key=self.output_key,
                          decomposition_path=self.decomposition_path,
                          dependency=t_1)
        return t_2

    def get_config():
        configs = super(DecompositionWorkflow, DecompositionWorkflow).get_config()
        configs.update({'solve_subproblems': subproblem_tasks.SolveSubproblemsLocal.default_task_config(),
                        'decompose': decompose_tasks.DecomposeLocal.default_task_config(),
                        'insert': insert_tasks.InsertLocal.default_task_config()})
        return configs
