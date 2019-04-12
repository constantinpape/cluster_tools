from xml.dom import minidom
import zipfile
# import csv
# import os

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


def write_n5(ds, skel_id, skel_vol, coordinate_offset=None):
    """ Write skeleton to custom n5-based format

    The skeleton data is stored via varlen chunks: each chunk contains
    the data for one skeleton and stores:
    [n_skel_points, coord_z_0, coord_y_0, coord_x_0, ..., coord_z_n, coord_y_n, coord_x_n,
     n_edges, edge_0_u, edge_0_v, ..., edge_n_u, edge_n_v]

    Arguments:
        ds [z5py.Dataset]: output dataset
        skel_id [int]: id of the object corresponding to the skeleton
        skel_vol [np.ndarray]: binary volume containing the skeleton
        coordinate_offset [listlike]: offset to coordinate (default: None)
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
    # check if we have offset and add up if we do
    if coordinate_offset is not None:
        assert len(coordinate_offset) == 3
        coords += coordinate_offset

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
# nml / nmx parser
#
# based on:
# https://github.com/knossos-project/knossos_utils/blob/master/knossos_utils/skeleton.py


def parse_attributes(xml_elem, parse_input):
    """ Parse xml input:

    Arguments:
        xml_elem: an XML parsing element containing an "attributes" member
        parse_input: [["attribute_name", python_type_name],
                      ["52", int], ["1.234", float], ["neurite", str], ...]
    Returns:
        list of python-typed values - [52, 1.234, "neurite", ...]
    """
    parse_output = []
    attributes = xml_elem.attributes
    for x in parse_input:
        try:
            if x[1] == int:
                # ensure float strings can be parsed too
                parse_output.append(int(float(attributes[x[0]].value)))
            else:
                parse_output.append(x[1](attributes[x[0]].value))
        except (KeyError, ValueError):
            parse_output.append(None)
    return parse_output


# Construct annotation. Annotations are trees (called things inside the nml files).
def read_coords_from_nml(nml):
    annotation_elems = nml.getElementsByTagName("thing")
    skeleton_coordinates = {}

    # TODO parse the skeleton id and use as key instead of linear index
    for skel_id, annotation_elem in enumerate(annotation_elems):
        node_elems = annotation_elem.getElementsByTagName("node")
        coords = []
        for node_elem in node_elems:
            x, y, z = parse_attributes(node_elem,
                                       [['x', int], ['y', int], ['z', int]])
            # TODO is this stored in physical coordinates?
            # need to transform appropriately
            coords.append([z, y, x])
        skeleton_coordinates[skel_id] = coords
    #
    return skeleton_coordinates


# TODO read and return tree structure, comments etc.
def parse_nml(nml_str):
    # TODO figure this out
    # read the pixel size
    # try:
    #     param = nml_str.getElementsByTagName("parameters")[0].getElementsByTagName("scale")[0]
    #     file_scaling = parse_attributes(param,
    #                                     [["x", float], ["y", float], ["z", float]])
    # except IndexError:
    #     # file_scaling = [1, 1, 1]
    #     pass
    coord_dict = read_coords_from_nml(nml_str)
    return coord_dict


# TODO return additional annotations etc
# TODO figure out scaling
def read_nml(input_path):
    """ Read skeleton stored in nml or nmx format

    NML format used by Knossos
    For details on the nml format see .

    Arguments:
        input_path [str]: path to swc file
    """
    # from knossos zip
    if input_path.endswith('k.zip'):
        zipper = zipfile.ZipFile(input_path)
        if 'annotation.xml' not in zipper.namelist():
            raise Exception("k.zip file does not contain annotation.xml")
        xml_string = zipper.read('annotation.xml')
        nml = minidom.parseString(xml_string)
        out = parse_nml(nml)

    # from nmx (pyKnossos)
    elif input_path.endswith('nmx'):

        out = {}
        with zipfile.ZipFile(input_path, 'r') as zf:
            for ff in zf.namelist():
                if not ff.endswith('.nml'):
                    continue
                nml = minidom.parseString(zf.read(ff))
                out[ff] = parse_nml(nml)

    # from nml
    else:
        nml = minidom.parse(input_path)
        out = parse_nml(nml)

    return out


def write_nml():
    pass

#
# csv parser
#


def read_csv():
    pass


def write_csv():
    pass
