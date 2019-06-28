import os
import ctypes


def set_numpy_threads(n_threads):
    """ Set the number of threads numpy exposes to its
    underlying linalg library. 

    This needs to be called BEFORE the numpy import and sets the number
    of threads statically.
    Based on answers in https://github.com/numpy/numpy/issues/11826.
    """

    # set number of threads for mkl if it is used
    try:
        import mkl
        mkl.set_num_threaads(n_threads)
    except Exception:
        pass

    for name in ['libmkl_rt.so', 'libmkl_rt.dylib', 'mkl_Rt.dll']:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(n_threads)))
        except Exception:
            pass

    # set number of threads in all possibly relevant environment variables
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_NUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
