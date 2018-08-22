import os
import json
import subprocess


def grid_search():
    # use_dts = [False, True]
    # use_lmc = [False, True]
    ws_type = 'ws_thresh'
    use_lmc = 0
    use_rf = 0
    weight_mc_edges = False
    weight_merge_edges = False
    block_id = '2'
    for weight_mc in (0, 1):
        for weight_merge in (0, 1):
            subprocess.call(['python', 'single_run.py', block_id, ws_type,
                             str(use_rf), str(use_lmc),
                             str(weight_mc), str(weight_merge)])


if __name__ == '__main__':
    grid_search()
