import torch
from torch import nn


def conv2d(param, adaptation, meta):
    w = nn.Parameter(torch.ones(param['out-channels'], param['in-channels'], param['kernal'], param['kernal']))
    torch.nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(param['out-channels']))
    w.meta, b.meta = meta, meta
    w.adaptation, b.adaptation = adaptation, adaptation
    return w, b


def linear(out_dim, in_dim, adaptation, meta):
    w = nn.Parameter(torch.ones(out_dim, in_dim))
    torch.nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(out_dim))
    w.meta, b.meta = meta, meta
    w.adaptation, b.adaptation = adaptation, adaptation
    return w, b


def batch_norm(param):
    w = nn.Parameter(torch.ones(param['in-channels']))
    # must set requires_grad=False
    running_mean = nn.Parameter(torch.zeros(param['in-channels']), requires_grad=False)
    running_var = nn.Parameter(torch.ones(param['in-channels']), requires_grad=False)
    return w, running_mean, running_var
