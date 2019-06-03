import luigi
from ..cluster_tasks import WorkflowBase
from ..node_labels import NodeLabelWorkflow
from . import predict as predict_tasks
from . import probs_to_costs as transform_tasks


class EdgeCostsWorkflow(WorkflowBase):

    features_path = luigi.Parameter()
    features_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    node_label_dict = luigi.DictParameter(default={})
    seg_path = luigi.Parameter(default=None)
    seg_key = luigi.Parameter(default=None)
    rf_path = luigi.Parameter(default='')

    def _costs_with_rf(self, dep, mapped_label_dict):
        predict_task = getattr(predict_tasks,
                               self._get_task_name('Predict'))
        dep = predict_task(tmp_folder=self.tmp_folder,
                           max_jobs=self.max_jobs,
                           config_dir=self.config_dir,
                           rf_path=self.rf_path,
                           features_path=self.features_path,
                           features_key=self.features_key,
                           output_path=self.output_path,
                           output_key=self.output_key,
                           dependency=dep)
        transform_task = getattr(transform_tasks,
                                 self._get_task_name('ProbsToCosts'))
        dep = transform_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             input_path=self.output_path,
                             input_key=self.output_key,
                             features_path=self.features_path,
                             features_key=self.features_key,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             dependency=dep,
                             node_label_dict=mapped_label_dict)
        return dep

    def _costs(self, dep, mapped_label_dict):
        transform_task = getattr(transform_tasks,
                                 self._get_task_name('ProbsToCosts'))
        dep = transform_task(tmp_folder=self.tmp_folder,
                             max_jobs=self.max_jobs,
                             config_dir=self.config_dir,
                             input_path=self.features_path,
                             input_key=self.features_key,
                             features_path=self.features_path,
                             features_key=self.features_key,
                             output_path=self.output_path,
                             output_key=self.output_key,
                             dependency=dep,
                             node_label_dict=mapped_label_dict)
        return dep

    def _map_node_labels(self, dep):
        assert self.seg_path is not None and self.seg_key is not None
        mapped_label_dict = {}
        for mode, pk in self.node_label_dict.items():
            path, key = pk
            prefix = 'cost_labels_%s' % '_'.join(key.split('/'))
            out_key = '%s_mapped' % key
            dep = NodeLabelWorkflow(tmp_folder=self.tmp_folder, config_dir=self.config_dir,
                                    target=self.target, max_jobs=self.max_jobs, dependency=dep,
                                    ws_path=self.seg_path, ws_key=self.seg_key,
                                    input_path=path, input_key=key,
                                    prefix=prefix,
                                    output_path=path, output_key=out_key,
                                    max_overlap=True, ignore_label=None)
            mapped_label_dict[mode] = (path, out_key)
        return dep, mapped_label_dict

    def requires(self):
        dep = self.dependency
        # if we have a node label dict, we need to do the mappings
        if self.node_label_dict:
            dep, mapped_label_dict = self._map_node_labels(dep)
        else:
            mapped_label_dict = None
        if self.rf_path == '':
            return self._costs(dep, mapped_label_dict)
        else:
            return self._costs_with_rf(dep, mapped_label_dict)

    @staticmethod
    def get_config():
        configs = super(EdgeCostsWorkflow, EdgeCostsWorkflow).get_config()
        configs.update({'probs_to_costs':
                        transform_tasks.ProbsToCostsLocal.default_task_config()})
        return configs
