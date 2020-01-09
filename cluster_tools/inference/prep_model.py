try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def extract_unet(model):
    return model.unet


def add_sigmoid(model):
    if torch is None:
        raise RuntimeError("Need torch to add sigmoid to model")
    n_out = getattr(model, 'out_channels', None)
    model = nn.Sequential(model, nn.Sigmoid())
    model.n_out = n_out
    return model


PREP_FUNCTIONS = {'extract_unet': extract_unet,
                  'add_sigmoid': add_sigmoid}


def get_prep_model(key):
    assert key in PREP_FUNCTIONS, "prep_model %s is not supported, use one of %s"\
        % (key, str(list(PREP_FUNCTIONS.values())))
    return PREP_FUNCTIONS[key]
