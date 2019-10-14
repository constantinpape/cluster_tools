import os
from functools import partial
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

try:
    from neurofire.inference.test_time_augmentation import TestTimeAugmenter
except ImportError:
    TestTimeAugmenter = None


#
# Prediction classes
#

class PytorchPredicter(object):

    @staticmethod
    def build_augmenter(augmentation_mode, augmentation_dim):
        return TestTimeAugmenter.default_tda(augmentation_dim, augmentation_mode)

    def __init__(self, model_path, halo, gpu=0, use_best=True, prep_model=None,
                 **augmentation_kwargs):
        # load the model and prep it if specified
        assert os.path.exists(model_path), model_path
        self.model = torch.load(model_path)
        self.model.eval()
        self.gpu = gpu
        self.model.cuda(self.gpu)
        if prep_model is not None:
            self.model = prep_model(self.model)
        #
        self.halo = halo
        self.lock = threading.Lock()
        # build the test-time-augmenter if we have augmentation kwargs
        if augmentation_kwargs:
            assert TestTimeAugmenter is not None, "Need neurofire for test-time-augmentation"
            self.offsets = augmentation_kwargs.pop('offsets', None)
            self.augmenter = self.build_augmenter(**augmentation_kwargs)
        else:
            self.augmenter = None

    def crop(self, out):
        shape = out.shape if out.ndim == 3 else out.shape[1:]
        bb = tuple(slice(ha, sh - ha) for ha, sh in zip(self.halo, shape))
        if out.ndim == 4:
            bb = (slice(None),) + bb
        return out[bb]

    def apply_model(self, input_data):
        with self.lock, torch.no_grad():
            if isinstance(input_data, np.ndarray):
                torch_data = torch.from_numpy(input_data[None, None]).cuda(self.gpu)
            else:
                torch_data = [torch.from_numpy(d[None, None]).cuda(self.gpu) for d in input_data]
            out = self.model(torch_data)
            # we send the data
            if torch.is_tensor(out):
                out = out.cpu().numpy().squeeze()
            elif isinstance(out, (list, tuple)):
                out = [o.cpu().numpy().squeeze() for o in out]
            else:
                raise TypeError("Expect model output to be tensor or list of tensors, got %s" % type(out))
        return out

    def apply_model_with_augmentations(self, input_data):
        out = self.augmenter(input_data, self.apply_model, self.offsets)
        return out

    def check_data(self, data):
        if isinstance(data, np.ndarray):
            assert data.ndim == 3
        elif isinstance(data, (list, tuple)):
            assert all(isinstance(d, np.ndarray) for d in data)
            assert all(d.ndim == 3 for d in data)
        else:
            raise ValueError("Need array or list of arrays")

    def __call__(self, input_data):
        self.check_data(input_data)
        if self.augmenter is None:
            out = self.apply_model(input_data)
        else:
            out = self.apply_model_with_augmentations(input_data)
        out = self.crop(out)
        return out


class InfernoPredicter(PytorchPredicter):
    def __init__(self, model_path, halo, gpu=0, use_best=True, prep_model=None, **augmentation_kwargs):
        # load the model and prep it if specified
        assert os.path.exists(model_path), model_path
        trainer = Trainer().load(from_directory=model_path, best=use_best)
        self.model = trainer.model.cuda(gpu)
        self.model.eval()
        if prep_model is not None:
            self.model = prep_model(self.model)
        self.gpu = gpu
        self.halo = halo
        self.lock = threading.Lock()

        # build the test-time-augmenter if we have augmentation kwargs
        if augmentation_kwargs:
            assert TestTimeAugmenter is not None, "Need neurofire for test-time-augmentation"
            self.offsets = augmentation_kwargs.pop('offsets', None)
            print(augmentation_kwargs)
            self.augmenter = self.build_augmenter(**augmentation_kwargs)
        else:
            self.augmenter = None


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


def normalize01(data, eps=1e-4):
    min_ = data.min()
    max_ = data.max()
    return (data - min_) / (max_ + eps)


def cast(data, dtype='float32'):
    return data.astype(dtype, copy=False)


def preprocess_torch(data, mean=None, std=None,
                     use_zero_mean_unit_variance=True):
    normalizer = partial(normalize, mean=mean, std=std)\
        if use_zero_mean_unit_variance else normalize01
    if isinstance(data, np.ndarray):
        data = normalizer(cast(data))
    elif isinstance(data, (list, tuple)):
        data = [normalizer(cast(d)) for d in data]
    else:
        raise ValueError("Invalid type %s" % type(data))
    return data


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
