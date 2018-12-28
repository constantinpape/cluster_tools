import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from .. import write as write_tasks
from . import reduce_lifted_problem as reduce_tasks
from . import solve_lifted_global as solve_tasks
from . import solve_lifted_subproblems as subproblem_tasks


class LiftedMulticutWorkflowBase(WorkflowBase):
    problem_path = luigi.Parameter()
    n_scales = luigi.IntParameter()
    lifted_prefix = luigi.Parameter()  # parameter for lifted problem prefix

    # tasks for the hierarchical solver solutions
    def _hierarchical_tasks(self, dependency, n_scales):
        subproblem_task = getattr(subproblem_tasks,
                                  self._get_task_name('SolveLiftedSubproblems'))
        reduce_task = getattr(reduce_tasks,
                              self._get_task_name('ReduceLiftedProblem'))
        dep = dependency
        for scale in range(n_scales):
            dep = subproblem_task(tmp_folder=self.tmp_folder,
                                  max_jobs=self.max_jobs,
                                  config_dir=self.config_dir,
                                  problem_path=self.problem_path,
                                  lifted_prefix=self.lifted_prefix,
                                  scale=scale,
                                  dependency=dep)
            dep = reduce_task(tmp_folder=self.tmp_folder,
                              max_jobs=self.max_jobs,
                              config_dir=self.config_dir,
                              problem_path=self.problem_path,
                              lifted_prefix=self.lifted_prefix,
                              scale=scale,
                              dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(LiftedMulticutWorkflowBase, LiftedMulticutWorkflowBase).get_config()
        configs.update({'solve_lifted_subproblems':
                        subproblem_tasks.SolveLiftedSubproblemsLocal.default_task_config(),
                        'reduce_lifted_problem':
                        reduce_tasks.ReduceLiftedProblemLocal.default_task_config()})
        return configs


class LiftedMulticutWorkflow(LiftedMulticutWorkflowBase):
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()

    def requires(self):
        solve_task = getattr(solve_tasks,
                             self._get_task_name('SolveLiftedGlobal'))
        dep = self._hierarchical_tasks(self.dependency, self.n_scales)
        t_solve = solve_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             problem_path=self.problem_path,
                             assignment_path=self.assignment_path,
                             assignment_key=self.assignment_key,
                             scale=self.n_scales,
                             lifted_prefix=self.lifted_prefix,
                             dependency=dep)
        return t_solve

    @staticmethod
    def get_config():
        configs = super(LiftedMulticutWorkflow, LiftedMulticutWorkflow).get_config()
        configs.update({'solve_lifted_global': solve_tasks.SolveLiftedGlobalLocal.default_task_config()})
        return configs
