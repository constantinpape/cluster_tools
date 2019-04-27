import os
import luigi
import z5py
import h5py
import nifty.distributed as ndist


class WriteCarving(luigi.Task):
    """ Write graph and features in carving format and other meta-data
    """
    input_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    features_key = luigi.Parameter()
    output_path = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def _write_graph(self):
        graph_path = os.path.join(self.input_path, self.graph_key)
        graph = ndist.Graph(graph_path)
        uv_ids = graph.uvIds()

        # TODO postprocess graph
        ilastk_graph_key = 'preprocessing/graph'
        with h5py.File(self.output_path) as f:
            # TODO compress ?
            f.create_dataset(ilastk_graph_key, data=uv_ids)

    def _write_features(self):
        f = z5py.File(self.input_path, 'r')
        ds = f[self.features_key]
        feats = ds[:, 0]

        # TODO postprocess features
        ilastk_feat_key = 'preprocessing/graph/edgeWeights'
        with h5py.File(self.output_path) as f:
            # TODO compress ?
            f.create_dataset(ilastk_feat_key, data=feats)

    def _write_metadata(self):
        with h5py.File(self.output_path) as f:
            # name of the workflow
            f.create_dataset('workflowName', data='Carving')
            # ilastik version
            f.create_dataset('ilastikVersion', data='1.3.0b2')
            # current applet: set to carving applet TODO find correct number
            f.create_dataset('currentApplet', data=2)
            # TODO what else do we need ?

    def run(self):
        self._write_graph()
        self._write_features()
        self._write_metadata()

    def output(self):
        return luigi.LocalTarget(self.output_path)
