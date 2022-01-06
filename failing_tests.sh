# Expected failures
# python test/features/test_edge_features.py TestEdgeFeatures.test_boundary_features
# python test/features/test_edge_features.py TestEdgeFeatures.test_affinity_features
# These failures might be due to luigi param errors
python test/mutex_watershed/test_mws_with_mask.py TestMwsWithMask.test_mws_with_mask
python test/watershed/test_watershed_without_mask.py TestWatershedWithoutMask.test_ws_2d
python test/watershed/test_watershed_without_mask.py TestWatershedWithoutMask.test_ws_3d
exit 1
