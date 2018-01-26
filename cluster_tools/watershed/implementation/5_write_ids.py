import os
import argparse
import nifty
import numpy as np


def assign_node_ids(block_id):
    block = blocking.getBlock(block_id)
    bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
    subvol = ds_out[bb]
    offset = offsets[block_id]
    subvol += offset
    ds_out[bb] = nifty.tools.take(node_labeling, subvol)


def watershed_step5():
    pass


if __name__ == '__main__':
    watershed_step5()
