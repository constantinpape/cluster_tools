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

    def __init__(self, model_path, halo, gpu=0, use_best=True, prep_model=None, **augmentation_kwargs):
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
            torch_data = torch.from_numpy(input_data[None, None]).cuda(self.gpu)
            predicted_on_gpu = self.model(torch_data)
            if isinstance(predicted_on_gpu, tuple):
                predicted_on_gpu = predicted_on_gpu[0]
            out = predicted_on_gpu.cpu().numpy().squeeze()
        return out

    def apply_model_with_augmentations(self, input_data):
        out = self.augmenter(input_data, self.apply_model, self.offsets)
        return out

    def __call__(self, input_data):
        assert isinstance(input_data, np.ndarray)
        assert input_data.ndim == 3
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
    return normalizer(cast(data))


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
