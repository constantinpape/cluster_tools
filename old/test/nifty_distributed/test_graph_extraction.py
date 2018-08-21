import unittest
import os
import nifty.distributed as ndist


class TestGraphExtraction(unittest.TestCase):
    path_to_graph = '/home/papec/Work/my_projects/cluster_tools/prototype/test/graph.n5/graph'
    path_to_nodes = '/home/papec/Work/my_projects/cluster_tools/prototype/test/nodes_to_blocks.n5/node_'
    graph_block_prefix = '/home/papec/Work/my_projects/cluster_tools/prototype/test/graph.n5/sub_graphs/s0/block_'

    def test_extraction(self):

        # load complete graph
        graph = ndist.loadAsUndirectedGraph(self.path_to_graph)

        # TODO test for more block-ids
        block_ids = list(range(64))
        for block_id in block_ids:
            block_path = self.graph_block_prefix + str(block_id)
            assert os.path.exists(block_path), block_path
            nodes = ndist.loadNodes(block_path)
            if len(nodes) == 1:
                continue

            # get the subgraph from in-memory graph
            inner_edges_a, outer_edges_a, graph_a = graph.extractSubgraphFromNodes(nodes)

            # get the subgraph from disc
            inner_edges_b, outer_edges_b, uvs_b = ndist.extractSubgraphFromNodes(nodes,
                                                                                 self.path_to_nodes,
                                                                                 self.graph_block_prefix)
            # tests for equality
            n_nodes_a = graph_a.numberOfNodes
            uvs_a = graph_a.uvIds()
            n_nodes_b = int(uvs_b.max() + 1)
            # test graph
            self.assertEqual(n_nodes_a, n_nodes_b)
            self.assertEqual(uvs_a.shape, uvs_b.shape)
            self.assertTrue((uvs_a == uvs_b).all())

            # test edge ids
            self.assertEqual(inner_edges_a.shape, inner_edges_b.shape)
            self.assertTrue((inner_edges_a == inner_edges_b).all())

            self.assertEqual(outer_edges_a.shape, outer_edges_b.shape)
            self.assertTrue((outer_edges_a == outer_edges_b).all())


if __name__ == '__main__':
    unittest.main()
