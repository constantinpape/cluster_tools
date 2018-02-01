import argparse
import numpy as np
import nifty.distributed as ndist


def graph_step4(graph_path, scale, block_file):
    block_list = np.load(block_file)
    input_key = 'graph'
    ndist.mapEdgeIds(graph_path, input_key,
                     blockGroup='sub_graphs/s%s' % scale,
                     blockPrefix="block_",
                     blockList=block_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("--block_file", type=str)
    args = parser.parse_args()

    graph_step4(args.graph_path, args.last_scale, args.block_file)
