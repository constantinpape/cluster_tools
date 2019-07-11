import numpy as np
import nifty
from skimage.measure import marching_cubes_lewiner
try:
    from marching_cubes import march
except ImportError:
    march is None


# speed up ?
def smooth_mesh(verts, normals, faces, iterations):
    n_verts = len(verts)
    g = nifty.graph.undirectedGraph(n_verts)

    edges = np.concatenate([faces[:, :2],
                            faces[:, 1:],
                            faces[:, ::2]], axis=0)
    g.insertEdges(edges)

    current_verts = verts
    current_normals = normals
    new_verts = np.zeros_like(verts, dtype=verts.dtype)
    new_normals = np.zeros_like(normals, dtype=normals.dtype)

    for it in range(iterations):
        for vert in range(n_verts):
            nbrs = np.array([vert] + [nbr[0] for nbr in g.nodeAdjacency(vert)],
                            dtype='int')
            new_verts[vert] = np.mean(current_verts[nbrs], axis=0)
            new_normals[vert] = np.mean(current_normals[nbrs], axis=0)
        current_verts = new_verts
        current_normals = new_normals

    return new_verts, new_normals


def marching_cubes(obj, smoothing_iterations=0, use_ilastik=False):
    if use_ilastik:
        if march is None:
            raise RuntimeError("Ilastik marching cubes implementation not found")
        verts, normals, faces = march(obj.T, smoothing_iterations)
        verts = verts[:, ::-1]
    else:
        verts, faces, normals, _ = marching_cubes_lewiner(obj)
        if smoothing_iterations > 0:
            verts, normals = smooth_mesh(verts, normals, faces, smoothing_iterations)

    return verts, faces, normals


def read_numpy(path):
    mesh = np.load(path)
    return mesh['verts'], mesh['faces'], mesh['normals']


def write_numpy(path, verts, faces, normals):
    np.savez_compressed(path,
                        verts=verts,
                        faces=faces,
                        normals=normals)


# TODO support different format for faces
def read_obj(path):
    verts = []
    faces = []
    normals = []
    face_normals = []
    with open(path) as f:
        for line in f:
            # normal
            if line.startswith('vn'):
                normals.append([float(ll) for ll in line.split()[1:]])
            # vertex texture, hard-coded to vt 0.0 0.0 in paintera
            elif line.startswith('vt'):
                pass
            # vertex
            elif line.startswith('v'):
                verts.append([float(ll) for ll in line.split()[1:]])
            # face
            elif line.startswith('f'):
                faces.append([int(ll.split('/')[0]) for ll in line.split()[1:]])
                face_normals.append([int(ll.split('/')[2]) for ll in line.split()[1:]])

    return (np.array(verts), np.array(faces),
            np.array(normals), np.array(face_normals))


# TODO support different format for faces
def write_obj(path, verts, faces, normals, face_normals):
    with open(path, 'w') as f:
        for vert in verts:
            f.write(" ".join(map(str, ['v'] + vert.tolist())))
            f.write("\n")

        f.write("\n")

        for normal in normals:
            f.write(" ".join(map(str, ['vn'] + normal.tolist())))
            f.write("\n")

        f.write("\n")
        f.write("vt 0.0 0.0\n")
        f.write("\n")

        for face, normal in zip(faces, face_normals):
            f.write(" ".join(["f"] + ["/".join([str(fa), "1", str(no)])
                                      for fa, no in zip(face, normal)]))
            f.write("\n")
