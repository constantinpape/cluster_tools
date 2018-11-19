import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import solve_subproblems as subproblem_tasks
from . import reduce_problem as reduce_tasks
from . import solve_global as solve_tasks
from . import sub_solutions as sub_tasks


class MulticutWorkflowBase(WorkflowBase):
    problem_path = luigi.Parameter()
    n_scales = luigi.IntParameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

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

    def get_config():
        configs = super(MulticutWorkflowBase, MulticutWorkflowBase).get_config()
        configs.update({'solve_subproblems': subproblem_tasks.SolveSubproblemsLocal.default_task_config(),
                        'reduce_problem': reduce_tasks.ReduceProblemLocal.default_task_config()})
        return configs


class MulticutWorkflow(MulticutWorkflowBase):

    def requires(self):
        solve_task = getattr(solve_tasks,
                             self._get_task_name('SolveGlobal'))
        dep = self._hierarchical_tasks(self.dependency, self.n_scales)
        t_solve = solve_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             problem_path=self.problem_path,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             scale=self.n_scales,
                             dependency=dep)
        return t_solve

    def get_config():
        configs = super(MulticutWorkflow, MulticutWorkflow).get_config()
        configs.update({'solve_global': solve_tasks.SolveGlobalLocal.default_task_config()})
        return configs


class SubSolutionsWorkflow(MulticutWorkflowBase):
    ws_path = luigi.Parameter()
    ws_key = luigi.Parameter()
    roi_begin = luigi.ListParameter(default=None)
    roi_end = luigi.ListParameter(default=None)

    def requires(self):
        sub_task = getattr(sub_tasks,
                           self._get_task_name('SubSolutions'))
        dep = self._hierarchical_tasks(self.dependency, self.n_scales + 1)
        t_sub = sub_task(tmp_folder=self.tmp_folder,
                         max_jobs=self.max_jobs,
                         config_dir=self.config_dir,
                         problem_path=self.problem_path,
                         ws_path=self.ws_path,
                         ws_key=self.ws_key,
                         output_path=self.output_path,
                         output_key=self.output_key,
                         scale=self.n_scales,
                         dependency=dep,
                         roi_begin=self.roi_begin,
                         roi_end=self.roi_end)
        return t_sub

    def get_config():
        configs = super(SubSolutionsWorkflow, SubSolutionsWorkflow).get_config()
        configs.update({'sub_solutions': sub_tasks.SubSolutionsLocal.default_task_config()})
        return configs
