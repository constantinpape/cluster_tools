import os
# import csv
import numpy as np
from skan import csr

# TODO implement converter between formats


#
# swc parser
#

def read_swc(input_path, return_radius=False, return_type=False):
    """ Read skeleton stored in .swc

    For details on the swc format for skeletons, see
    http://research.mssm.edu/cnic/swc.html.
    This function expects the swc catmaid flavor.

    Arguments:
        input_path [str]: path to swc file
        retun_radius [bool]: return radius measurements (default: False)
        retun_type [bool]: return type variable (default: False)
    """
    ids, coords, parents = [], [], []
    radii, types = [], []
    # open file and get outputs
    with open(input_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            # skip headers or break
            if line.startswith('#') or line == '':
                continue

            # parse this line
            values = line.split()
            # extract coordinate, node-id and parent-id
            coords.append([float(val) for val in values[2:5]])
            ids.append(int(values[0]))
            parents.append(int(values[-1]))

            # extract radius
            if return_radius:
                radii.append(float(values[5]))

            # extract type
            if return_type:
                types.append(int(values[1]))

    if return_radius:
        return ids, coords, parents, radii
    if return_type:
        return ids, coords, parents, types
    if return_radius and return_type:
        return ids, coords, parents, radii, types
    return ids, coords, parents


def write_swc(output_path, skel_vol, resolution=None, invert_coords=False):
    """ Write skeleton to .swc

    For details on the swc format for skeletons, see
    http://research.mssm.edu/cnic/swc.html.
    This writes the swc catmaid flavor.

    Arguments:
        output_path [str]: output_path for swc file
        skel_vol [np.ndarray]: binary volume containing the skeleton
        resolution [list or float]: pixel resolution (default: None)
        invert_coords [bool]: whether to invert the coordinates
            This may be useful because swc expects xyz, but input is zyx (default: False)
    """
    # extract the skeleton graph
    # NOTE looks like skan function names are about to change in 0.8:
    # csr.numba_csgraph -> csr.csr_to_nbgraph

    # this may fail for small skeletons with a value-error
    try:
        pix_graph, coords, _ = csr.skeleton_to_csgraph(skel_vol)
    except ValueError:
        return
    graph = csr.numba_csgraph(pix_graph)

    # map coords to resolution and invert if necessary
    if resolution is not None:
        if isinstance(resolution, float):
            resolution = 3 * [resolution]
        assert len(resolution) == 3, str(len(resolution))
        coords *= resolution
    if invert_coords:
        coords = coords[:, ::-1]

    # TODO if this becomes a bottle-neck think about moving to numba, cython or c++
    n_points = pix_graph.shape[0]
    with open(output_path, 'w') as f:
        for node_id in range(1, n_points):
            # swc: node-id
            #      type (hard-coded to 0 = undefined)
            #      coordinates
            #      radius (hard-coded to 0.0)
            #      parent id
            ngbs = graph.neighbors(node_id)

            # only a single neighbor -> terminal node and no parent
            # also, for some reasons ngbs can be empty
            if len(ngbs) in (0, 1):
                parent = -1
            # two neighbors -> path node
            # more than two neighbors -> junction
            else:
                # TODO can we just assume that we get consistent output if we set parent to min ???
                parent = np.min(ngbs)
            coord = coords[node_id]
            line = '%i 0 %f %f %f 0.0 %i \n' % (node_id, coord[0], coord[1], coord[2], parent)
            f.write(line)


#
# n5 parser
#


def read_n5(ds, skel_id):
    """ Read skeleton stored in custom n5-based format

    The skeleton data is stored via varlen chunks: each chunk contains
    the data for one skeleton and stores:
    (n_skel_points, coord_z_0, coord_y_0, coord_x_0, ..., coord_z_n, coord_y_n, coord_x_n,
     n_edges, edge_0_u, edge_0_v, ..., edge_n_u, edge_n_v)

    Arguments:
        ds [z5py.Dataset]: input dataset
        skel_id [int]: id of the object corresponding to the skeleton
    """
    # read data from chunk
    data = ds.read_chunk((skel_id,))

    # check if the chunk is empty
    if data is None:
        return None, None

    # read number of points and coordinates
    n_points = data[0]
    offset = 1
    coord_len = int(3 * n_points)
    coords = data[offset:offset+coord_len].reshape((n_points, 3))
    offset += coord_len
    # read number of edges and edges
    n_edges = data[offset]
    offset += 1
    edge_len = int(2 * n_edges)
    assert len(data) == offset + edge_len, "%i, %i" % (len(data), offset + edge_len)
    edges = data[offset:offset+edge_len].reshape((n_edges, 2))
    return coords, edges


def write_n5(ds, skel_id, skel_vol):
    """ Write skeleton to custom n5-based format

    The skeleton data is stored via varlen chunks: each chunk contains
    the data for one skeleton and stores:
    [n_skel_points, coord_z_0, coord_y_0, coord_x_0, ..., coord_z_n, coord_y_n, coord_x_n,
     n_edges, edge_0_u, edge_0_v, ..., edge_n_u, edge_n_v]

    Arguments:
        ds [z5py.Dataset]: output dataset
        skel_id [int]: id of the object corresponding to the skeleton
        skel_vol [np.ndarray]: binary volume containing the skeleton
    """
    # NOTE looks like skan function names are about to change in 0.8:
    # csr.numba_csgraph -> csr.csr_to_nbgraph
    # extract the skeleton graph

    # this may fail for small skeletons with a value-error
    try:
        pix_graph, coords, _ = csr.skeleton_to_csgraph(skel_vol)
    except ValueError:
        return
    graph = csr.numba_csgraph(pix_graph)

    # skan-indexing is 1 based, so we need to get rid of first coordinate row
    coords = coords[1:]
    # make serialization for number of points and coordinates
    n_points = coords.shape[0]
    data = [np.array([n_points]), coords.flatten()]

    # make edges
    edges = [[u, v] for u in range(1, n_points + 1) for v in graph.neighbors(u) if u < v]
    edges = np.array(edges)
    # substract 1 to change to zero-based indexing
    edges -= 1
    # add number of edges and edges to the serialization
    n_edges = len(edges)
    data.extend([np.array([n_edges]), edges.flatten()])

    data = np.concatenate(data, axis=0)
    ds.write_chunk((skel_id,), data.astype('uint64'), True)


#
# csv parser
#


def read_csv():
    pass


def write_csv():
    pass
