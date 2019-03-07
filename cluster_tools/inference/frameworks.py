import os
import threading
import numpy as np

# try to load the frameworks
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow
except ImportError:
    tensorflow = None

try:
    from inferno.trainers.basic import Trainer
except ImportError:
    Trainer = None


#
# Prediction classes
#

class PytorchPredicter(object):
    def __init__(self, model_path, halo, gpu=0, prep_model=None):
        assert os.path.exists(model_path), model_path
        self.model = torch.load(model_path)
        self.model.eval()
        self.gpu = gpu
        self.model.cuda(self.gpu)
        if prep_model is not None:
            self.model = prep_model(self.model)
        self.halo = halo
        self.lock = threading.Lock()

    def crop(self, out):
        shape = out.shape if out.ndim == 3 else out.shape[1:]
        bb = tuple(slice(ha, sh - ha) for ha, sh in zip(shape, shape))
        if out.ndim == 4:
            bb = (slice(None),) + bb
        return out[bb]

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray)
        assert input_data.ndim == 3
        # Note: in the code that follows, the GPU is locked for the 3 steps:
        # CPU -> GPU, GPU inference, GPU -> CPU. It may well be that we get
        # better performance by only locking in step 2, or steps 1-2, or steps
        # 2-3. We should perform this experiment and then choose the best
        # option for our hardware (and then remove this comment! ;)
        with self.lock, torch.no_grad():
            # 1. Transfer the data to the GPU
            torch_data = torch.from_numpy(input_data[None, None]).cuda(self.gpu)
            # 2. Run the model
            predicted_on_gpu = self.model(torch_data)
            if isinstance(predicted_on_gpu, tuple):
                predicted_on_gpu = predicted_on_gpu[0]
            # 3. Transfer the results to the CPU
            out = predicted_on_gpu.cpu().numpy().squeeze()
        out = self.crop(out)
        return out


class InfernoPredicter(PytorchPredicter):
    def __init__(self, model_path, halo, gpu=0, use_best=True, prep_model=None):
        assert os.path.exists(model_path), model_path
        trainer = Trainer().load(from_directory=model_path, best=use_best)
        self.model = trainer.model.cuda(gpu)
        self.model.eval()
        if prep_model is not None:
            self.model = prep_model(self.model)
        self.gpu = gpu
        self.halo = halo
        self.lock = threading.Lock()


# TODO
class TensorflowPredicter(object):
    pass


def get_predictor(framework):
    if framework == 'pytorch':
        assert torch is not None
        return PytorchPredicter
    elif framework == 'inferno':
        assert torch is not None
        assert Trainer is not None
        return InfernoPredicter
    elif framework == 'tensorflow':
        assert tensorflow is not None
        return TensorflowPredicter
    else:
        raise KeyError("Framework %s not supported" % framework)


#
# Pre-processing functions
#

def normalize(data, eps=1e-4, mean=None, std=None, filter_zeros=True):
    if filter_zeros:
        data_pre = data[data != 0]
    else:
        data_pre = data
    mean = data_pre.mean() if mean is None else mean
    std = data_pre.std() if std is None else std
    return (data - mean) / (std + eps)


def cast(data, dtype='float32'):
    return data.astype(dtype, copy=False)


def preprocess_torch(data, mean=None, std=None):
    return normalize(cast(data), mean=mean, std=std)


# TODO
def preprocess_tf():
    pass


def get_preprocessor(framework):
    if framework in ('inferno', 'pytorch'):
        return preprocess_torch
    elif framework == 'tensorflow':
        return preprocess_tf
    else:
        raise KeyError("Framework %s not supported" % framework)
