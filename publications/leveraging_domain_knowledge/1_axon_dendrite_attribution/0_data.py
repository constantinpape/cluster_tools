import argparse
import h5py
import z5py


def copy_input(input_path):
    output_path = './data.n5'
    with h5py.File(input_path, 'r') as f_in, z5py.File(output_path, 'a') as f_out:

        def copy_ds(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            print("Copying", name)
            data = obj[:]
            ds = f_out.create_dataset(name, shape=data.shape, chunks=(70, 70, 70),
                                      compression='gzip', dtype=data.dtype)
            ds.n_threads = 4
            ds[:] = data

        f_in.visititems(copy_ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_path = '/home/pape/Work/data/mmwc/knott_data.h5'
    parser.add_argument('--input_path', type=str, default=default_path)
    args = parser.parse_args()
    copy_input(args.input_path)
