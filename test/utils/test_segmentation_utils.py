import sys
import os
import unittest
from shutil import rmtree

import nifty
import nifty.graph.opt.multicut as nmc
import numpy as np

try:
    import cluster_tools
except ImportError:
    sys.path.append('../..')
    import cluster_tools


class TestSegmentationUtil(unittest.TestCase):
    problem_root = '/home/cpape/Work/data/multicut_problems'
    problem = 'small/small_problem_sampleA.txt'
    path = os.path.join(problem_root, problem)
    upper_bound = -76900

    @staticmethod
    def load_problem(path):
        edges = []
        costs = []
        with open(path, 'r') as f:
            for l in f:
                r = l.split()
                edges.append([int(r[0]), int(r[1])])
                costs.append(float(r[2]))
        edges = np.array(edges, dtype='uint64')
        edges = np.sort(edges, axis=1)
        costs = np.array(costs)
        n_nodes = int(edges.max()) + 1
        graph = nifty.graph.undirectedGraph(n_nodes)
        graph.insertEdges(edges)
        return graph, costs

    # TODO return expeceted edges
    @staticmethod
    def toy_problem():
        graph = nifty.graph.undirectedGraph(10)
        edges = np.array([[0, 1],  # 0
                          [0, 2],  # 1
                          [0, 3],  # 2
                          [1, 2],  # 3
                          [1, 4],  # 4
                          [1, 7],  # 5
                          [2, 6],  # 6
                          [3, 6],  # 7
                          [4, 5],  # 8
                          [4, 8],  # 9
                          [4, 9],  # 10
                          [5, 6],  # 11
                          [7, 8],  # 12
                          [8, 9]   # 13
                          ], dtype='uint64')
        graph.insertEdges(edges)
        costs = np.array([-1, -3, 6, -3, -2, 3, 2, -1, -2, -1, 3, 1, 4, 1])
        assert len(costs) == len(edges)
        return graph, costs

    def _check_result(self, graph, costs, node_labels):
        self.assertEqual(graph.numberOfNodes, len(node_labels))
        obj = nmc.multicutObjective(graph, costs)
        energy = obj.evalNodeLabels(node_labels)
        self.assertGreater(self.upper_bound, energy)
        return energy

    def test_mc_kl(self):
        from cluster_tools.utils.segmentation_utils import multicut_kernighan_lin
        graph, costs = self.load_problem(self.path)
        node_labels = multicut_kernighan_lin(graph, costs)
        energy = self._check_result(graph, costs, node_labels)
        print("kernighan-lin:", energy)

    def test_mc_gaec(self):
        from cluster_tools.utils.segmentation_utils import multicut_gaec
        graph, costs = self.load_problem(self.path)
        node_labels = multicut_gaec(graph, costs)
        energy = self._check_result(graph, costs, node_labels)
        print("gaec:", energy)

    def test_mc_decompose(self):
        from cluster_tools.utils.segmentation_utils import multicut_decomposition
        graph, costs = self.load_problem(self.path)
        node_labels = multicut_decomposition(graph, costs)
        energy = self._check_result(graph, costs, node_labels)
        print("decomposition:", energy)

    def test_decompose_toy(self):
        from cluster_tools.utils.segmentation_utils import multicut_decomposition
        from cluster_tools.utils.segmentation_utils import multicut_kernighan_lin
        graph, costs = self.toy_problem()
        # node_labels = multicut_decomposition(graph, costs)
        node_labels = multicut_kernighan_lin(graph, costs)
        print(node_labels)


if __name__ == '__main__':
    unittest.main()
