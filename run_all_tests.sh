# In some cases luigi does not correctly update task parameters when the same task, but with different input parameters
# is run in the same python process, as is the case when using unittest discover
# hence we need to run the tests "by hand" here and exit if any of the tests fails

# For some reason these tests also need to be run all separately
python test/connected_components/connected_components.py TestConnectedComponents.test_greater
if [[ $? != 0 ]]
then
    exit 1
fi
python test/connected_components/connected_components.py TestConnectedComponents.test_less
if [[ $? != 0 ]]
then
    exit 1
fi
python test/connected_components/connected_components.py TestConnectedComponents.test_equal
if [[ $? != 0 ]]
then
    exit 1
fi

# need to run test separately due to some failed cleanup
python test/downscaling/test_downscaling.py TestDownscaling.test_downscaling_paintera
if [[ $? != 0 ]]
then
    exit 1
fi
python test/downscaling/test_downscaling.py TestDownscaling.test_downscaling_bdv_h5
if [[ $? != 0 ]]
then
    exit 1
fi
python test/downscaling/test_downscaling.py TestDownscaling.test_downscaling_bdv_n5
if [[ $? != 0 ]]
then
    exit 1
fi
python test/downscaling/test_downscaling.py TestDownscaling.test_downscaling_ome_zarr
if [[ $? != 0 ]]
then
    exit 1
fi
python test/downscaling/test_downscaling.py TestDownscaling.test_downscaling_int_to_uint
if [[ $? != 0 ]]
then
    exit 1
fi

python test/evaluation/test_evaluation.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/features/test_edge_features.py
if [[ $? != 0 ]]
then
    exit 1
fi
python test/features/test_region_features.py
if [[ $? != 0 ]]
then
    exit 1
fi

# for the graph test both of the tests in the same unittest need to be run separately
python test/graph/test_graph.py TestGraph.test_graph
if [[ $? != 0 ]]
then
    exit 1
fi
python test/graph/test_graph.py TestGraph.test_graph_label_multiset
if [[ $? != 0 ]]
then
    exit 1
fi

python test/ilastik/test_pixel_classification.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/inference/test_inference.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/label_multisets/test_label_multisets.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/lifted_features/lifted_features.py
if [[ $? != 0 ]]
then
    exit 1
fi
python test/lifted_features/sparse_lifted_neighborhood.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/morphology/test_morphology.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/mutex_watershed/test_mws.py
if [[ $? != 0 ]]
then
    exit 1
fi
python test/mutex_watershed/test_mws_with_mask.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/node_labels/test_node_labels.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/postprocess/test_postprocess.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/relabel/test_relabel.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/retry/test_retry.py
if [[ $? != 0 ]]
then
    exit 1
fi

# Test is taking very long
# python test/skeletons/test_skeletons.py
# if [[ $? != 0 ]]
# then
#     exit 1
# fi

python test/statistics/test_statistics.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/transformations/test_linear.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/utils/test_function_utils.py
if [[ $? != 0 ]]
then
    exit 1
fi
python test/utils/test_function_utils.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/watershed/test_watershed_with_mask.py
if [[ $? != 0 ]]
then
    exit 1
fi
python test/watershed/test_watershed_without_mask.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/workflows/lifted_multicut_workflow.py
if [[ $? != 0 ]]
then
    exit 1
fi
python test/workflows/multicut_workflow.py
if [[ $? != 0 ]]
then
    exit 1
fi

python test/write/test_write.py
if [[ $? != 0 ]]
then
    exit 1
fi
