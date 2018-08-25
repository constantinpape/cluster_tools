import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import solve_subproblems as subproblem_tasks
from . import reduce_problem as reduce_tasks
from . import solve_global as solve_tasks


class MulticutWorkflow(WorkflowBase):
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    costs_path = luigi.Parameter()
    costs_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    n_scales = luigi.Parameter()

    def requires(self):

        subproblem_task = getattr(subproblem_tasks,
                                  self._get_task_name('SolveSubproblems'))
        reduce_task = getattr(reduce_tasks,
                              self._get_task_name('ReduceProblem'))
        solve_task = getattr(solve_tasks,
                             self._get_task_name('SolveGlobal'))

        t_prev = self.dependency
        # TODO input path should change at different scales !
        for scale in range(self.n_scales):
            t_sub = subproblem_task(tmp_folder=self.tmp_folder,
                                    max_jobs=self.max_jobs,
                                    config_dir=self.config_dir,
                                    graph_path=self.graph_path,
                                    graph_key=self.graph_key,
                                    costs_path=self.costs_path,
                                    costs_key=self.costs_key,
                                    scale=scale,
                                    dependency=t_prev)
            t_reduce = reduce_task(tmp_folder=self.tmp_folder,
                                   max_jobs=self.max_jobs,
                                   config_dir=self.config_dir,
                                   graph_path=self.graph_path,
                                   graph_key=self.graph_key,
                                   costs_path=self.costs_path,
                                   costs_key=self.costs_key,
                                   scale=scale,
                                   dependency=t_prev)
            t_prev = t_reduce

        t_solve = solve_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             self.output_path,
                             self.output_key,
                             scale=self.n_scales - 1,
                             dependency=t_prev)
        return t_solve
