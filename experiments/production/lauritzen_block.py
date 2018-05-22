import os
import json
import luigi
from production import Workflow, write_dt_components_config, write_default_config


def process_lauritzen_block(block_id,
                            n_jobs, n_threads,
                            use_dt_components=False,
                            use_lmc=False,
                            weight_mc_edges=False,
                            weight_merge_edges=False):

    cache_prefix = '/groups/saalfeld/home/papec/Work/neurodata_hdd/cache/'
    cache_name_ws = 'lauritzen_production_%i_ws' % block_id
    cache_name_ws += '_dt' if use_dt_components else '_thresh'
    cache_folder_ws = os.path.join(cache_prefix, cache_name_ws)

    cache_name_mc = 'lauritzen_production_%i' % block_id
    mc_string = 'lmc' if use_lmc else 'mc'
    mc_string += '_dt' if use_dt_components else '_thresh'
    mc_string += '_mcweighted' if weight_mc_edges else '_nomcweighted'
    mc_string += '_mergeweighted' if weight_merge_edges else '_nomergeweighted'
    cache_folder_mc = os.path.join(cache_prefix, cache_name_mc + '_' + mc_string)

    if use_dt_components:
        boundary_threshold = .15
    else:
        boundary_threshold = .05

    if use_lmc:
        lifted_nh = 2
        lifted_rf_path = os.path.join('/groups/saalfeld/home/papec/Work/my_projects/cluster_tools/experiments',
                                      'production/lifted_rf_larger_nh.pkl')
    else:
        lifted_nh = 2
        lifted_rf_path = None

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
                         lifted_rf_path=lifted_rf_path,
                         weighting_exponent=weighting_exponent,
                         weight_multicut_edges=weight_mc_edges,
                         weight_merge_edges=weight_merge_edges,
                         n_threads=n_threads)
    if use_dt_components:
        write_dt_components_config(config_path,
                                   boundary_threshold=boundary_threshold)

    ws_key = 'ws_dt' if use_dt_components else 'ws_thresh'
    seg_key = mc_string

    # skeleton_keys = ['skeletons/for_eval_20180508',
    #                  'skeletons/neurons_of_interest']
    skeleton_keys = []
    path = '/nrs/saalfeld/lauritzen/0%i/workspace.n5' % block_id
    luigi.run(['--local-scheduler',
               '--path', path,
               '--aff-key', 'raw/predictions/affs_glia',
               '--mask-key', 'raw/masks/minfilter_mask',
               '--ws-key', 'raw/segmentation/' + ws_key,
               '--seg-key', 'raw/segmentation/' + seg_key,
               '--max-jobs', str(n_jobs),
               '--config-path', config_path,
               '--tmp-folder-ws', cache_folder_ws,
               '--tmp-folder-seg', cache_folder_mc,
               '--skeleton-keys', json.dumps(skeleton_keys),
               '--time-estimate', '10'], Workflow)


def grid_search():
    # use_dts = [False, True]
    # use_lmc = [False, True]
    use_dt = False
    use_lmc = True
    weight_mc_edges = False
    weight_merge_edges = False
    process_lauritzen_block(2, 300, 12,
                            use_dt_components=use_dt,
                            use_lmc=use_lmc,
                            weight_mc_edges=weight_mc_edges,
                            weight_merge_edges=weight_merge_edges)


if __name__ == '__main__':
    grid_search()
