#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import z5py
import luigi
from cluster_tools.postprocess import FilterOrphansWorkflow
from cluster_tools.postprocess import SizeFilterAndGraphWatershedWorkflow


# FIXME this does not work yet
def filter_orphans(input_path):
    """ Filter orphaned segments (= segments with only a single neighbor)
    """
    exp_path = './sampleA_exp.n5'

    tmp_folder = './tmp_orphans'
    config_folder = './configs'

    task = FilterOrphansWorkflow(tmp_folder=tmp_folder, config_dir=config_folder,
                                 target='local', max_jobs=8,
                                 graph_path=exp_path, graph_key='s0/graph',
                                 path=input_path, assignment_key='node_labels',
                                 segmentation_key='volumes/segmentation/watershed',
                                 output_path=input_path,
                                 assignment_out_key='node_labels_orphans',
                                 output_key='volumes/segmentation/filtered_orphans')
    luigi.build([task], local_scheduler=True)


def graph_watershed_size_filter(input_path):
    """ Filter small segments
    """
    exp_path = './sampleA_exp.n5'

    tmp_folder = './tmp_ws_filter'
    config_folder = './configs'

    task_name = SizeFilterAndGraphWatershedWorkflow
    task = task_name(tmp_folder=tmp_folder, config_dir=config_folder,
                     target='local', max_jobs=4,
                     problem_path=exp_path, graph_key='s0/graph',
                     features_key='features',
                     path=input_path, assignment_key='node_labels',
                     segmentation_key='volumes/segmentation/multicut',
                     fragments_key='volumes/segmentation/watershed',
                     output_path=input_path,
                     assignment_out_key='node_labels_graphws',
                     output_key='volumes/segmentation/filtered_graphws',
                     size_threshold=1000, relabel=True)
    ret = luigi.build([task], local_scheduler=True)
    assert ret


def check_results(input_path):
    from cremi_tools.viewer.volumina import view
    f = z5py.File(input_path)

    ds = f['volumes/raw/s0']
    ds.n_threads = 8
    raw = ds[:]

    ds = f['volumes/segmentation/multicut']
    ds.n_threads = 8
    seg1 = ds[:]

    # ds = f['volumes/segmentation/filtered_orphans']
    # ds.n_threads = 8
    # seg2 = ds[:]

    ds = f['volumes/segmentation/filtered_graphws']
    ds.n_threads = 8
    seg3 = ds[:]

    # view([raw, seg1, seg2, seg3],
    #      ['raw', 'mc', 'orphans', 'graph-ws'])
    view([raw, seg1, seg3],
         ['raw', 'mc', 'graph-ws'])


if __name__ == '__main__':
    input_path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    graph_watershed_size_filter(input_path)
    check_results(input_path)
