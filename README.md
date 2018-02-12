# Cluster Tools

Workflows for 3D Neuron-EM-segmentation on the Janelia Cluster.

## Workflows

- Connected Components
- Watersheds
- Masked Watersheds
- Region Graph
- Edge Features from Boundary or Affinity Maps
- Block-wise Agglomeration (Multicut or Agglomerative Clustering)

TODO: Masked Watersheds:
- Move call to  `prepare.py` before call to `make_batch_jobs.py`.
- Check blocking for blocks different from chunks.
- The resulting segmentation is not consecutive, because we have 'dead' ids
  in the overlaps. I am sure there is a way to circumvent this , but it is not
  trivial. For now, do a relabeling instead.
