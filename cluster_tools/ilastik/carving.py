import os
import numpy as np
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
    raw_path = luigi.Parameter(default=None)
    raw_key = luigi.Parameter(default=None)
    copy_inputs = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def _write_graph(self):
        graph_path = os.path.join(self.input_path, self.graph_key)
        graph = ndist.Graph(graph_path)
        uv_ids = graph.uvIds().astype('uint32')

        # the ilastik graph serialization corresponds to a the serialization of
        # vigra:adjacencyListGraph, see
        # https://github.com/constantinpape/vigra/blob/master/include/vigra/adjacency_list_graph.hxx#L536

        # first element: node and edge numbers, max node and edge ids
        n_nodes = graph.numberOfNodes
        n_edges = graph.numberOfEdges
        serialization = [np.array([n_nodes, n_edges,
                                   n_nodes - 1, n_edges - 1], dtype='uint32')]
        # second element: uv-ids
        serialization.append(uv_ids.flatten())
        # third element: node neighborhoods (implemented convinience function in cpp for this)
        serialization.append(graph.flattenedNeighborhoods().astype('uint32'))
        serialization = np.concatenate(serialization)

        ilastk_graph_key = 'preprocessing/graph/graph'
        ilastik_seed_key = 'preprocessing/graph/nodeSeeds'
        ilastik_res_key = 'preprocessing/graph/resultSegmentation'
        with h5py.File(self.output_path) as f:
            f.create_dataset(ilastk_graph_key, data=serialization,
                             compression='gzip')
            # initialize node seed labels and result labels with zeros
            f.create_dataset(ilastik_seed_key, shape=(n_nodes,), dtype='uint8')
            f.create_dataset(ilastik_res_key, shape=(n_nodes,), dtype='uint8')

    def _write_features(self):
        f = z5py.File(self.input_path, 'r')
        ds = f[self.features_key]
        feats = ds[:, 0]

        # carving features have val-range 0 - 255
        feats *= 255

        ilastk_feat_key = 'preprocessing/graph/edgeWeights'
        with h5py.File(self.output_path) as f:
            f.create_dataset(ilastk_feat_key, data=feats,
                             compression='gzip')

    # TODO adapt for copy_inputs = True
    def _write_input_metadata(self):
        with h5py.File(self.output_path) as f:
            g = f.create_group('Input Data/infos/lane0000/Raw Data')
            g.create_dataset('allowLabels', data=True)
            g.create_dataset('axisorder', data='zyx')
            # PLEASE PLEASE PLEASE let this be optional
            # g.create_dataset('axistags', data=self.default_axis_tags
            # g.create_dataset('datasetId', data=)
            g.create_dataset('display_mode', data='default')

            # TODO implement
            if self.copy_inputs:
                pass
            else:
                assert self.raw_path is not None
                assert self.raw_key is not None
                path = os.path.join(self.raw_path, self.raw_key)
                g.create_dataset('filePath', data=path)
                g.create_dataset('location', data='fileSystem')
            g.create_dataset('nickname', data='Input')

    def _write_metadata(self):
        with h5py.File(self.output_path) as f:
            # general metadata
            # name of the workflow
            f.create_dataset('workflowName', data='Carving')
            # ilastik version
            f.create_dataset('ilastikVersion', data='1.3.0b2')
            # current applet: set to carving applet TODO find correct number
            f.create_dataset('currentApplet', data=2)

            # preprocessing metadata
            f.create_dataset('preprocessing/StorageVersion', data='0.1')
            # filter id (set to gaussian) and sugma
            f.create_dataset('preprocessing/filter', data=3)
            f.create_dataset('preprocessing/sigma', data=1.)
            # don't know what these values mean and if they are even still necessary
            f.create_dataset('preprocessing/invert_watershed_source', data=False)
            f.create_dataset('preprocessing/watershed_source', data='filtered')

    def run(self):
        self._write_graph()
        self._write_features()
        self._write_input_metadata()
        self._write_metadata()

    def output(self):
        return luigi.LocalTarget(self.output_path)
