from concurrent import futures
import z5py
import numpy as np
from .io import read_edge_result


def oracle(seg_id, ds_in,
           ds_uvs, ds_costs, oracle_precision, oracle_recall,
           max_attractive, max_repulsive):
    uvs, indicators = read_edge_result(ds_in, seg_id)
    if uvs is None:
        return
    # print("Processing", seg_id, "/", ds_in.shape[0])

    # oracle recall < 1:
    # we switch merge indications with the p = oracle_recall
    if oracle_recall < 1.:
        original_indicators = indicators.copy()
        n_merges = indicators.sum()
        indicators[indicators] = np.random.rand(n_merges) < oracle_recall
    else:
        original_indicators = indicators

    # oracle precision < 1:
    # we switch non-merge indications with the p = oracle_precision
    if oracle_precision < 1.:
        # we use 'original_indicators' here in order not to switch
        # back decisions from oracle_recall
        non_indicators = np.logical_not(original_indicators)
        n_non_merges = non_indicators.sum()
        indicators[non_indicators] = np.random.rand(n_non_merges) > oracle_precision

    # don't add edges if no merge is detected
    n_merges = indicators.sum()
    if n_merges == 0:
        return

    costs = np.zeros(len(uvs), dtype='float32')

    # oracle recall < 1:
    # weight the repulsive edges with p = oracle_recall
    if oracle_recall < 1:
        weights = (1. - oracle_recall) * np.random.rand(n_merges) + oracle_recall
        costs[indicators] = max_repulsive * weights
    else:
        costs[indicators] = max_repulsive

    # oracle precision < 1:
    # weight the attractive edges with p = oracle_precision
    non_indicators = np.logical_not(indicators)
    n_non_merges = non_indicators.sum()
    if oracle_precision < 1:
        weights = (1. - oracle_precision) * np.random.rand(n_non_merges) + oracle_precision
        costs[np.logical_not(indicators)] = max_attractive * weights
    else:
        costs[np.logical_not(indicators)] = max_attractive

    ds_uvs.write_chunk((seg_id,), uvs.flatten(), True)
    ds_costs.write_chunk((seg_id,), costs, True)


# compute costs from lifted edges with (inperfect) oracle
def edges_to_problem(in_path, out_path,
                     key_in, key_out_uv, key_out_costs,
                     oracle_precision, oracle_recall,
                     max_attractive, max_repulsive,
                     n_threads):
    f_in = z5py.File(in_path)
    f_out = z5py.File(out_path)
    ds_in = f_in[key_in]

    n_labels = ds_in.shape[0]
    ds_uvs = f_out.require_dataset(key_out_uv, shape=(n_labels,), chunks=(1,),
                                   dtype='uint64', compression='gzip')
    ds_costs = f_out.require_dataset(key_out_costs, shape=(n_labels,), chunks=(1,),
                                     dtype='float32', compression='gzip')

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(oracle, seg_id, ds_in,
                           ds_uvs, ds_costs, oracle_precision, oracle_recall,
                           max_attractive, max_repulsive) for seg_id in range(n_labels)]
        [t.result() for t in tasks]


# combine edges and costs from multiple objects to big lifted problem
def combine_edges_and_costs(in_path, in_group,
                            out_path, out_key_uvs, out_key_costs,
                            n_threads):
    f = z5py.File(in_path)
    group = f[in_group]

    print(group.path)
    ds_uvs = group['uvs']
    ds_costs = group['costs']
    n_objects = ds_costs.shape[0]

    def load_obj(obj_id):
        uvs = ds_uvs.read_chunk((obj_id,))
        if uvs is None:
            return None
        n_edges = len(uvs) // 2
        uvs = uvs.reshape((n_edges, 2))
        costs = ds_costs.read_chunk((obj_id,))
        assert len(costs) == n_edges
        return uvs, costs

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(load_obj, obj_id)
                 for obj_id in range(n_objects)]
        results = [t.result() for t in tasks]

    results = [res for res in results if res is not None]

    uvs = np.concatenate([res[0] for res in results],
                         axis=0)
    costs = np.concatenate([res[1] for res in results],
                           axis=0)

    n_lifted = len(uvs)
    assert len(costs) == n_lifted
    print("Extracted %i lifted edges" % n_lifted)

    # numpy lexsort is very akward, see
    # https://stackoverflow.com/questions/38277143/sort-2d-numpy-array-lexicographically
    sorted_ids = np.lexsort(np.rot90(uvs))
    uvs = uvs[sorted_ids]
    costs = costs[sorted_ids]

    f = z5py.File(out_path)
    chunk_len = min(64**3, n_lifted)

    ds_uvs_out = f.require_dataset(out_key_uvs, shape=uvs.shape,
                                   compression='gzip', dtype='uint64',
                                   chunks=(chunk_len, 2))
    ds_uvs_out.n_threads = n_threads
    ds_uvs_out[:] = uvs

    ds_costs_out = f.require_dataset(out_key_costs, shape=costs.shape,
                                     compression='gzip', dtype='float32',
                                     chunks=(chunk_len,))
    ds_costs_out.n_threads = n_threads
    ds_costs_out[:] = costs
