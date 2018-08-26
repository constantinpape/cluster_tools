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
    n_scales = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    merged_problem_path = luigi.Parameter()

    def requires(self):

        subproblem_task = getattr(subproblem_tasks,
                                  self._get_task_name('SolveSubproblems'))
        reduce_task = getattr(reduce_tasks,
                              self._get_task_name('ReduceProblem'))
        solve_task = getattr(solve_tasks,
                             self._get_task_name('SolveGlobal'))

        t_prev = self.dependency
        for scale in range(self.n_scales):
            # graph and costs path change with the scales !
            if scale == 0:
                graph_path = self.graph_path
                graph_key = self.graph_key
                costs_path = self.costs_path
                costs_key = self.costs_key
            else:
                graph_path = self.merged_problem_path
                graph_key = 's%i/graph' % scale
                costs_path = self.merged_problem_path
                costs_key = 's%i/costs' % scale
            t_sub = subproblem_task(tmp_folder=self.tmp_folder,
                                    max_jobs=self.max_jobs,
                                    config_dir=self.config_dir,
                                    graph_path=graph_path,
                                    graph_key=graph_key,
                                    costs_path=costs_path,
                                    costs_key=costs_key,
                                    scale=scale,
                                    dependency=t_prev)
            t_reduce = reduce_task(tmp_folder=self.tmp_folder,
                                   max_jobs=self.max_jobs,
                                   config_dir=self.config_dir,
                                   graph_path=graph_path,
                                   graph_key=graph_key,
                                   costs_path=costs_path,
                                   costs_key=costs_key,
                                   output_path=self.merged_problem_path,
                                   scale=scale,
                                   dependency=t_sub)
            t_prev = t_reduce

        t_solve = solve_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             input_path=self.merged_problem_path,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             scale=self.n_scales,
                             dependency=t_prev)
        return t_solve
