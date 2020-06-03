# Axon/Dendrite Attribution Experiments

This folder contains the code for the experiments for the lifted costs from axon/dendrite attribution.
There are two different versions of this workflow: the `full` version, that uses edge weights that are learned from
a random forest and the `simple` version, that uses edge weights that are directly estimated from probability maps.

In the paper, we have used the `full` version, because the probability maps were estimated only by a random forest and are hence not
good enought to be turned into edge costs directly.
If you want to try it on your own data, we suggest to start from the simple workflow. In the case of good probability maps from a CNN
it might be sufficient already

## Running the simple workflow

Run the scripts in the following order:
```
0_data.py
1_multicut_simple.py
2_lifted_multicut_simple.py
```

You can use `3_evaluation.py` to compute the segmentation metrics and `4_view_results.py` to visualize the results (needs napari).


## Running the full workflow

TODO not added yet
