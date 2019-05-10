#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

import z5py
import luigi
from cluster_tools.postprocess import FilterOrphansWorkflow
from cluster_tools.postprocess import SizeFilterAndGraphWatershedWorkflow
from cremi_tools.viewer.volumina import view


# TODO check results
def filter_orphans(sample='A'):
    input_path = '/g/kreshuk/data/cremi/example/sample%s.n5' % sample
    exp_path = './sample%s_exp.n5' % sample

    tmp_folder = './tmp_orphans'
    config_folder = './config_mc'

    task = FilterOrphansWorkflow(tmp_folder=tmp_folder, config_dir=config_folder,
                                 target='local', max_jobs=8,
                                 graph_path=exp_path, graph_key='s0/graph',
                                 path=input_path, assignment_key='node_labels',
                                 segmentation_key='segmentation/watershed',
                                 output_path=input_path,
                                 assignment_out_key='node_labels_orphans',
                                 output_key='segmentation/filtered_orphans')
    luigi.build([task], local_scheduler=True)


def graph_watershed_size_filter(sample='A'):
    input_path = '/g/kreshuk/data/cremi/example/sample%s.n5' % sample
    exp_path = './sample%s_exp.n5' % sample

    tmp_folder = './tmp_ws_filter'
    config_folder = './config_mc'

    task_name = SizeFilterAndGraphWatershedWorkflow
    task = task_name(tmp_folder=tmp_folder, config_dir=config_folder,
                     target='local', max_jobs=8,
                     problem_path=exp_path, graph_key='s0/graph',
                     features_key='features',
                     path=input_path, assignment_key='node_labels',
                     segmentation_key='segmentation/multicut',
                     fragments_key='segmentation/watershed',
                     output_path=input_path,
                     assignment_out_key='node_labels_graphws',
                     output_key='segmentation/filtered_graphws',
                     size_threshold=1000, relabel=True)
    ret = luigi.build([task], local_scheduler=True)
    assert ret


def check_results(sample='A'):
    input_path = '/g/kreshuk/data/cremi/example/sample%s.n5' % sample
    f = z5py.File(input_path)

    ds = f['raw']
    ds.n_threads = 8
    raw = ds[:]

    ds = f['segmentation/multicut']
    ds.n_threads = 8
    seg1 = ds[:]

    ds = f['segmentation/filtered_orphans']
    ds.n_threads = 8
    seg2 = ds[:]

    ds = f['segmentation/filtered_graphws']
    ds.n_threads = 8
    seg3 = ds[:]

    view([raw, seg1, seg2, seg3],
         ['raw', 'mc', 'orphans', 'graph-ws'])


if __name__ == '__main__':
    # filter_orphans()
    graph_watershed_size_filter()
    check_results()
