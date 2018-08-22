import os
import json
import argparse
import luigi
from production import Workflow, Workflow2DWS
from production import write_dt_components_config, write_default_config


def process_lauritzen_block(block_id,
                            n_jobs, n_threads,
                            ws_type='ws_thresh',
                            use_rf=False,
                            use_lmc=False,
                            weight_mc_edges=False,
                            weight_merge_edges=False):

    assert ws_type in ('ws_thresh', 'ws_2d', 'ws_dt')

    cache_prefix = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/'
    cache_name_ws = 'lauritzen_production_%i' % block_id
    cache_name_ws += ('_' + ws_type)
    cache_folder_ws = os.path.join(cache_prefix, cache_name_ws)

    cache_name_mc = 'lauritzen_production_%i' % block_id
    mc_string = 'lmc_' if use_lmc else 'mc_'
    mc_string += ws_type
    if use_rf:
        mc_string += '_rf'
    mc_string += '_mcweighted' if weight_mc_edges else '_nomcweighted'
    mc_string += '_mergeweighted' if weight_merge_edges else '_nomergeweighted'
    cache_folder_mc = os.path.join(cache_prefix, cache_name_mc + '_' + mc_string)

    if ws_type == 'ws_dt':
        boundary_threshold = .15
        distance_threshold = 50
    elif 'ws_type' == 'ws_2d':
        boundary_threshold = .2
    else:
        boundary_threshold = .05

    if use_lmc:
        lifted_nh = 2
        if use_rf:
            rf_path = [os.path.join('/groups/saalfeld/home/papec/Work/my_projects/cluster_tools/experiments',
                                    'production/lifted_rf_larger_nh.pkl')]
        else:
            rf_path = None
    else:
        lifted_nh = None
        if use_rf:
            rf_path = [os.path.join('/groups/saalfeld/home/papec/Work/my_projects/cluster_tools/experiments/production',
                                    'rf_local_%s.pkl' % etype) for etype in ('xy', 'z')]
        else:
            rf_path = None

    if weight_mc_edges:
        weighting_exponent = .5
    else:
        weighting_exponent = 1.

    try:
        os.mkdir(cache_folder_mc)
    except Exception:
        pass

    config_path = os.path.join(cache_folder_mc, 'config.json')
    write_default_config(config_path,
                         boundary_threshold=boundary_threshold,
                         lifted_nh=lifted_nh,
                         rf_path=rf_path,
                         weighting_exponent=weighting_exponent,
                         weight_multicut_edges=weight_mc_edges,
                         weight_merge_edges=weight_merge_edges,
                         n_threads=n_threads)
    if ws_type == 'ws_dt':
        write_dt_components_config(config_path,
                                   boundary_threshold=boundary_threshold,
                                   distance_threshold=distance_threshold)

    seg_key = mc_string

    skeleton_keys = ['skeletons/for_eval_20180523',
                     'skeletons/neurons_of_interest']
    # skeleton_keys = []
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    wf = Workflow2DWS if ws_type == 'ws_2d' else Workflow
    luigi.run(['--local-scheduler',
               '--path', path,
               '--aff-key', 'raw/predictions/affs_glia',
               '--mask-key', 'raw/masks/minfilter_mask',
               '--ws-key', 'raw/segmentation/' + ws_type,
               '--seg-key', 'raw/segmentation/' + seg_key,
               '--max-jobs', str(n_jobs),
               '--config-path', config_path,
               '--tmp-folder-ws', cache_folder_ws,
               '--tmp-folder-seg', cache_folder_mc,
               '--skeleton-keys', json.dumps(skeleton_keys),
               '--time-estimate', '10'], wf)


if __name__ == '__main__':
    n_jobs = 300
    n_threads = 12
    parser = argparse.ArgumentParser()
    parser.add_argument('block_id', type=int)
    parser.add_argument('ws_type', type=str)
    parser.add_argument('use_rf', type=int)
    parser.add_argument('use_lmc', type=int)
    parser.add_argument('weight_mc_edges', type=int)
    parser.add_argument('weight_merge_edges', type=int)
    args = parser.parse_args()

    process_lauritzen_block(args.block_id,
                            n_jobs, n_threads,
                            args.ws_type,
                            bool(args.use_rf), bool(args.use_lmc),
                            bool(args.weight_mc_edges), bool(args.weight_merge_edges))
