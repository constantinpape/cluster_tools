import os
import json
import luigi

from ..cluster_tasks import WorkflowBase
from . import initial_sub_graphs as initial_tasks
from . import merge_sub_graphs as merge_tasks
from . import map_edge_ids as map_tasks


class GraphWorkflow(WorkflowBase):
    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    output_key = luigi.Parameter()
    n_scales = luigi.IntParameter()

    # for now we only support n5 / zarr input labels
    def _check_input(self):
        ending = self.input_path.split('.')[-1]
        assert ending.lower() in ('zr', 'zarr', 'n5'),\
            "Only support n5 and zarr files, not %s" % ending

    def requires(self):
        self._check_input()

        initial_task = getattr(initial_tasks,
                               self._get_task_name('InitialSubGraphs'))
        t1 = initial_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          input_path=self.input_path,
                          input_key=self.input_key,
                          graph_path=self.graph_path,
                          dependency=self.dependency)
        merge_task = getattr(merge_tasks,
                             self._get_task_name('MergeSubGraphs'))
        t_prev = t1
        for scale in range(1, self.n_scales):
            t2 = merge_task(tmp_folder=self.tmp_folder,
                            max_jobs=self.max_jobs,
                            config_dir=self.config_dir,
                            graph_path=self.graph_path,
                            scale=scale,
                            merge_complete_graph=False,
                            dependency=t_prev)
            t_prev = t2

        t3 = merge_task(tmp_folder=self.tmp_folder,
                        max_jobs=self.max_jobs,
                        config_dir=self.config_dir,
                        graph_path=self.graph_path,
                        output_key=self.output_key,
                        scale=self.n_scales - 1,
                        merge_complete_graph=True,
                        dependency=t_prev)


        map_task = getattr(map_tasks,
                           self._get_task_name('MapEdgeIds'))
        t_prev = t3
        for scale in range(self.n_scales):
            t4 = map_task(tmp_folder=self.tmp_folder,
                          max_jobs=self.max_jobs,
                          config_dir=self.config_dir,
                          graph_path=self.graph_path,
                          input_key=self.output_key,
                          scale=self.n_scales - 1,
                          dependency=t_prev)
            t_prev = t4
        return t4

    @staticmethod
    def get_config():
        configs = super(GraphWorkflow, GraphWorkflow).get_config()
        configs.update({'initial_sub_graphs': initial_tasks.InitialSubGraphsLocal.default_task_config(),
                        'merge_sub_graphs': merge_tasks.MergeSubGraphsLocal.default_task_config(),
                        'map_edge_ids': map_tasks.MapEdgeIdsLocal.default_task_config()})
        return configs
