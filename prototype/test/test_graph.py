import sys
import os
from shutil import rmtree
import nifty.graph.rag as nrag
import z5py

sys.path.append('..')
from prototype import compute_region_graph, load_graph


def test_graph():
    seg_path = '/home/papec/Work/neurodata_hdd/ntwrk_papec/cluster_test_data/ws.n5'
    assert os.path.exists(seg_path)
    seg_key = 'data'
    blocks = (25, 256, 256)

    if os.path.exists('./graph.n5'):
        rmtree('./graph.n5')

    compute_region_graph(seg_path, seg_key, blocks, './graph.n5')
    nodes, edges = load_graph('./graph.n5', 'graph')

    labels = z5py.File(seg_path)[seg_key][:].astype('uint32')
    rag = nrag.gridRag(labels, numberOfLabels=labels.max()+1)
    assert rag.numberOfEdges == len(edges)
    assert (rag.uvIds() == edges).all()
    print("Passed!")

    # TODO check the sub-graph edge-ids !!


if __name__ == '__main__':
    test_graph()
