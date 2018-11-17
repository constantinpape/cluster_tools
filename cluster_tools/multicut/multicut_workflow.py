import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import solve_subproblems as subproblem_tasks
from . import reduce_problem as reduce_tasks
from . import solve_global as solve_tasks


class MulticutWorkflow(WorkflowBase):
    problem_path = luigi.Parameter()
    n_scales = luigi.IntParameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()

    def requires(self):

        subproblem_task = getattr(subproblem_tasks,
                                  self._get_task_name('SolveSubproblems'))
        reduce_task = getattr(reduce_tasks,
                              self._get_task_name('ReduceProblem'))
        solve_task = getattr(solve_tasks,
                             self._get_task_name('SolveGlobal'))

        t_prev = self.dependency
        for scale in range(self.n_scales):
            t_sub = subproblem_task(tmp_folder=self.tmp_folder,
                                    max_jobs=self.max_jobs,
                                    config_dir=self.config_dir,
                                    problem_path=self.problem_path,
                                    scale=scale,
                                    dependency=t_prev)
            t_reduce = reduce_task(tmp_folder=self.tmp_folder,
                                   max_jobs=self.max_jobs,
                                   config_dir=self.config_dir,
                                   problem_path=self.problem_path,
                                   scale=scale,
                                   dependency=t_sub)
            t_prev = t_reduce

        t_solve = solve_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             problem_path=self.problem_path,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             scale=self.n_scales,
                             dependency=t_prev)
        return t_solve

    def get_config():
        configs = super(MulticutWorkflow, MulticutWorkflow).get_config()
        configs.update({'solve_subproblems': subproblem_tasks.SolveSubproblemsLocal.default_task_config(),
                        'reduce_problem': reduce_tasks.ReduceProblemLocal.default_task_config(),
                        'solve_global': solve_tasks.SolveGlobalLocal.default_task_config()})
        return configs
