import numpy as np


def write_edge_result(ds, seg_id, lifted_uvs, merge_indicators):
    n_edges = len(lifted_uvs)
    data = [n_edges] + lifted_uvs.flatten().tolist() + merge_indicators.tolist()
    data = np.array(data, dtype='uint64')
    ds.write_chunk((seg_id,), data, True)


def read_edge_result(ds, seg_id):
    data = ds.read_chunk((seg_id,))
    if data is None:
        return None, None

    n_edges = int(data[0])
    offset = 1
    lifted_uvs = data[offset:(2*n_edges + offset)].reshape((n_edges, 2))
    offset += 2*n_edges
    merge_indicators = data[offset:].astype('bool')
    assert len(merge_indicators) == len(lifted_uvs) == n_edges
    return lifted_uvs, merge_indicators
