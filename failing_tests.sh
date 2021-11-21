# Expected failures
# python test/features/test_edge_features.py TestEdgeFeatures.test_boundary_features
# python test/features/test_edge_features.py TestEdgeFeatures.test_affinity_features
python test/graph/test_graph.py TestGraph.test_graph_label_multiset
python test/retry/test_retry.py TestRetry.test_retry
# These failures might be due to luigi param errors
python test/mutex_watershed/test_mws_with_mask.py TestMwsWithMask.test_mws_with_mask
pythin test/watershed/test_watershed_without_mask.py TestWatershedWithoutMask.test_ws_2d
pythin test/watershed/test_watershed_without_mask.py TestWatershedWithoutMask.test_ws_3d
exit 1
