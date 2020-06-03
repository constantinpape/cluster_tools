import napari
import z5py


def view_result():
    data_path = './data.n5'

    with z5py.File(data_path, 'r') as f:
        ds = f['raw']
        ds.n_threads = 4
        raw = ds[:]

        ds = f['probs/membranes']
        ds.n_threads = 4
        mem = ds[:]

        ds = f['volumes/segmentation/multicut']
        ds.n_threads = 4
        mc_seg = ds[:]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw, name='raw')
        viewer.add_image(mem, name='membranes')

        viewer.add_labels(mc_seg, name='mc-seg')


view_result()
