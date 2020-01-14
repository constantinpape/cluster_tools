import luigi

from ..cluster_tasks import WorkflowBase
from . import simple_stitch_edges as simple_stitch_tasks
from . import simple_stitch_assignments as stitch_assignment_tasks


class StitchingAssignmentsWorkflow(WorkflowBase):
    problem_path = luigi.Parameter()
    labels_path = luigi.Parameter()
    labels_key = luigi.Parameter()
    assignments_path = luigi.Parameter()
    assignments_key = luigi.Parameter()
    features_key = luigi.Parameter()
    graph_key = luigi.Parameter('s0/graph')
    edge_size_threshold = luigi.IntParameter(default=0)
    serialize_edges = luigi.BoolParameter(default=False)

    def requires(self):
        simple_stitch_task = getattr(simple_stitch_tasks,
                                     self._get_task_name('SimpleStitchEdges'))
        dep = simple_stitch_task(tmp_folder=self.tmp_folder,
                                 max_jobs=self.max_jobs,
                                 config_dir=self.config_dir,
                                 graph_path=self.problem_path,
                                 labels_path=self.labels_path,
                                 labels_key=self.labels_key,
                                 dependency=self.dependency)
        # from simple stitch edges to assignments
        stitch_assignment_task = getattr(stitch_assignment_tasks,
                                         self._get_task_name('SimpleStitchAssignments'))
        dep = stitch_assignment_task(tmp_folder=self.tmp_folder,
                                     max_jobs=self.max_jobs,
                                     config_dir=self.config_dir,
                                     problem_path=self.problem_path,
                                     graph_key=self.graph_key,
                                     features_key=self.features_key,
                                     edge_size_threshold=self.edge_size_threshold,
                                     serialize_edges=self.serialize_edges,
                                     assignments_path=self.assignments_path,
                                     assignments_key=self.assignments_key,
                                     dependency=dep)
        return dep

    @staticmethod
    def get_config():
        configs = super(StitchingAssignmentsWorkflow,
                        StitchingAssignmentsWorkflow).get_config()
        configs.update({'simple_stitch_edges':
                        simple_stitch_tasks.SimpleStitchEdgesLocal.default_task_config(),
                        'simple_stitch_assignments':
                        stitch_assignment_tasks.SimpleStitchAssignmentsLocal.default_task_config()})
        return configs
