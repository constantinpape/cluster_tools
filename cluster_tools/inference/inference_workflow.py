from ..cluster_tasks import WorkflowBase


# TODO
# inference workflow that enables block-pre-filtering:
# use ../block_filter/block_filter_workflow to eliminate the blocks without any data
# (if we have a mask), store in some binary format
# and then run the inference task only on the blocks with data
# motivation: setting up the dask graph for many blocks takes quite long, so for large dataset,
# where usually a large fraction of blocks does not contain any data, this can save time
class InferenceWorkflow(WorkflowBase):
    pass
