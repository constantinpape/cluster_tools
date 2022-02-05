import os
from contextlib import nullcontext
from functools import partial
from importlib import import_module

import threading
import numpy as np

# try to load the frameworks
try:
    import bioimageio.core
    from xarray import DataArray
except ImportError:
    bioimageio.core = None

try:
    import torch
except ImportError:
    torch = None

try:
    from inferno.trainers.basic import Trainer
except ImportError:
    Trainer = None

try:
    from neurofire.inference.test_time_augmentation import TestTimeAugmenter
except ImportError:
    TestTimeAugmenter = None


#
# Prediction Base class
#

class PredicterBase:
    def crop(self, out, halo):
        shape = out.shape if out.ndim == 3 else out.shape[1:]
        bb = tuple(slice(ha, sh - ha) for ha, sh in zip(halo, shape))
        if out.ndim == 4:
            bb = (slice(None),) + bb
        return out[bb]


#
# Pytorch based prediction classes
#

class PytorchPredicter(PredicterBase):

    @staticmethod
    def build_augmenter(augmentation_mode, augmentation_dim):
        return TestTimeAugmenter.default_tda(augmentation_dim, augmentation_mode)

    def set_up(self, halo, gpu, prep_model, mixed_precision, **augmentation_kwargs):
        self.model.eval()
        self.gpu = gpu
        self.model.cuda(self.gpu)
        if prep_model is not None:
            self.model = prep_model(self.model)

        self.mixed_precision = mixed_precision
        if self.mixed_precision:
            self.autocast = torch.cuda.amp.autocast
        else:
            self.autocast = nullcontext

        # save the halo and check if this is a multi-scale halo
        # (halo is nested list)
        self.halo = halo
        self.has_multiscale_halo = isinstance(halo[0], list)

        self.lock = threading.Lock()
        # build the test-time-augmenter if we have augmentation kwargs
        if augmentation_kwargs:
            assert TestTimeAugmenter is not None, "Need neurofire for test-time-augmentation"
            self.offsets = augmentation_kwargs.pop("offsets", None)
            self.augmenter = self.build_augmenter(**augmentation_kwargs)
        else:
            self.augmenter = None

    def load_model(self, model_path):
        # load model from a saved pytorch model
        if isinstance(model_path, str):
            assert os.path.exists(model_path), model_path
            model = torch.load(model_path)

        # load model from a model state dict
        elif isinstance(model_path, dict):
            module, model_name = model_path["class"]
            model_kwargs = model_path["kwargs"]
            model_class = getattr(import_module(module), model_name)
            model = model_class(**model_kwargs)

            state_path = model_path["checkpoint_path"]
            assert os.path.exists(state_path), state_path
            model_key = model_path.get("model_state_key", None)

            state = torch.load(state_path)
            if model_key is not None:
                state = state[model_key]
            model.load_state_dict(state)

        return model

    def __init__(self, model_path, halo, gpu=0, prep_model=None,
                 mixed_precision=False, use_best=True, **augmentation_kwargs):
        # load the model and prep it if specified
        self.model = self.load_model(model_path)
        self.set_up(halo, gpu, prep_model, mixed_precision,
                    **augmentation_kwargs)

    def apply_model(self, input_data):
        with self.lock, torch.no_grad(), self.autocast():
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
        if isinstance(out, list) and self.has_multiscale_halo:
            assert len(self.halo) == len(out) and all(isinstance(halo, list) for halo in self.halo)
            out = [self.crop(oo, halo) for oo, halo in zip(out, self.halo)]
        elif isinstance(out, list):
            out = out[0]
            out = self.crop(out, self.halo)
        else:
            out = self.crop(out, self.halo)
        return out


class InfernoPredicter(PytorchPredicter):
    def __init__(self, model_path, halo, gpu=0, use_best=True, prep_model=None,
                 mixed_precision=False, **augmentation_kwargs):
        # load the model and prep it if specified
        assert os.path.exists(model_path), model_path

        # this is left over from some hack to import modules that were not
        # available during import. Not a nice solution, but it works, so I am
        # leaving this here for reference
        # TimeTrainingIters = None

        self.model = Trainer().load(from_directory=model_path, best=use_best).model
        self.set_up(halo, gpu, prep_model, mixed_precision, **augmentation_kwargs)


#
# Bioimageio prediction calss
#


class BioimageioPredicter(PredicterBase):
    def __init__(self, model_path, halo, gpu=0, **kwargs):
        assert os.path.exists(model_path)
        model = bioimageio.core.load_resource_description(model_path)
        devices = [f"cuda:{gpu}"]
        self.model = bioimageio.core.create_prediction_pipeline(
            bioimageio_model=model, devices=devices
        )
        self.halo = halo
        self.lock = threading.Lock()

    def __del__(self):
        self.model.close()

    def __call__(self, input_data):
        input_ = DataArray(input_data, dims=self.dims)
        with self.lock:
            out = self.model(input_)
        assert len(out) == 1
        out = out[0].values
        out = self.crop(out, self.halo)
        return out


def get_predictor(framework):
    if framework == "pytorch":
        assert torch is not None
        return PytorchPredicter
    elif framework == "inferno":
        assert torch is not None
        assert Trainer is not None
        return InfernoPredicter
    elif framework == "bioimageio":
        assert bioimageio.core is not None
        return BioimageioPredicter
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


def cast(data, dtype="float32"):
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


def get_preprocessor(framework, **kwargs):
    if framework in ("inferno", "pytorch"):
        return partial(preprocess_torch, **kwargs)
    # bioimageio has preprocessing build in, so we return a no-op here
    elif framework == "bioimageio":
        return lambda x: x
    else:
        raise KeyError("Framework %s not supported" % framework)
