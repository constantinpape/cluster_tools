import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from .. import write as write_tasks
from . import reduce_problem as reduce_tasks
from . import solve_global as solve_tasks
from . import sub_solutions as sub_tasks


class LiftedMulticutWorkflowBase(WorkflowBase):
    problem_path = luigi.Parameter()
    n_scales = luigi.IntParameter()

    # tasks for the hierarchical solver solutions
    def _hierarchical_tasks(self, dependency, n_scales):
        subproblem_task = getattr(subproblem_tasks,
                                  self._get_task_name('SolveSubproblems'))
        reduce_task = getattr(reduce_tasks,
                              self._get_task_name('ReduceProblem'))
        dep = dependency
        for scale in range(n_scales):
            dep = subproblem_task(tmp_folder=self.tmp_folder,
                                  max_jobs=self.max_jobs,
                                  config_dir=self.config_dir,
                                  problem_path=self.problem_path,
                                  scale=scale,
                                  dependency=dep)
            dep = reduce_task(tmp_folder=self.tmp_folder,
                              max_jobs=self.max_jobs,
                              config_dir=self.config_dir,
                              problem_path=self.problem_path,
                              scale=scale,
                              dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(MulticutWorkflowBase, MulticutWorkflowBase).get_config()
        configs.update({'solve_subproblems':
                        subproblem_tasks.SolveSubproblemsLocal.default_task_config(),
                        'reduce_problem':
                        reduce_tasks.ReduceProblemLocal.default_task_config()})
        return configs


class LiftedMulticutWorkflow(MulticutWorkflowBase):
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()

    def requires(self):
        solve_task = getattr(solve_tasks,
                             self._get_task_name('SolveGlobal'))
        dep = self._hierarchical_tasks(self.dependency, self.n_scales)
        t_solve = solve_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             problem_path=self.problem_path,
                             assignment_path=self.assignment_path,
                             assignment_key=self.assignment_key,
                             scale=self.n_scales,
                             dependency=dep)
        return t_solve

    @staticmethod
    def get_config():
        configs = super(LiftedMulticutWorkflow, LiftedMulticutWorkflow).get_config()
        configs.update({'solve_global': solve_tasks.SolveGlobalLocal.default_task_config()})
        return configs
