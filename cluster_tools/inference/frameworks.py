import numpy as np

# try to load the frameworks
try:
    import torch
    WITH_TORCH = True
except ImportError:
    WITH_TORCH = False

try:
    import tensorflow
    WITH_TF = True
except ImportError:
    WITH_TF = False


def get_predictor(framework):
    pass


def get_preprocessor(framework):
    pass
