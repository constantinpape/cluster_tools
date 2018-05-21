import vigra
import numpy as np

import nifty
import nifty.graph.agglo as nagglo


def region_features(seg, uv_ids, input_):
    # FIXME for some reason 'Quantiles' are not working
    statistics = ["Mean", "Variance", "Skewness", "Kurtosis",
                  "Minimum", "Maximum", "Count", "RegionRadii"]
    extractor = vigra.analysis.extractRegionFeatures(input_, seg.astype('uint32'),
                                                     features=statistics)

    node_features = np.concatenate([extractor[stat_name][:, None].astype('float32')
                                    if extractor[stat_name].ndim == 1
                                    else extractor[stat_name].astype('float32')
                                    for stat_name in statistics],
                                   axis=1)
    fU = node_features[uv_ids[:, 0], :]
    fV = node_features[uv_ids[:, 1], :]

    edge_features = np.concatenate([np.minimum(fU, fV),
                                    np.maximum(fU, fV),
                                    np.abs(fU - fV)], axis=1)
    return np.nan_to_num(edge_features)


def ucm_features(n_labels, local_uvs, lifted_uvs, local_probs):

    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(local_uvs)
    graph.insertEdges(lifted_uvs)

    all_probs = np.concatenate([local_probs, np.zeros(len(lifted_uvs), dtype='float32')],
                               axis=0)

    return np.nan_to_num(nagglo.ucmFeatures(graph,
                                            all_probs,
                                            edgeSizes=None,
                                            nodeSizes=None)[len(local_uvs):])


def clustering_features(graph, probs, lifted_uvs):

    edge_sizes = np.ones(graph.numberOfEdges)
    node_sizes = np.ones(graph.numberOfNodes)

    def cluster(threshold):
        policy = nifty.graph.agglo.malaClusterPolicy(graph=graph,
                                                     edgeIndicators=probs,
                                                     edgeSizes=edge_sizes,
                                                     nodeSizes=node_sizes,
                                                     threshold=threshold)

        clustering = nifty.graph.agglo.agglomerativeClustering(policy)
        clustering.run()
        node_labels = clustering.result()
        return (node_labels[lifted_uvs[:, 0]] != node_labels[lifted_uvs[:, 1]]).astype('float32')

    thresholds = (.3, .4, .5, .6, .7, .8)
    features = np.concatenate([cluster(thresh)
                               for thresh in thresholds], axis=1)
    state_sum = np.sum(features, axis=1)[:, None]
    return np.concatenate([features, state_sum], axis=1)
