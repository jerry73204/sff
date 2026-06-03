"""Convenience API for the local `cuda` package.

NOTE: this package is a *namespace package* portion (no __init__.py) because the
installed `cuda-bindings` redirector (_cuda_bindings_redirector.pth) eagerly
imports the site-packages `cuda` namespace package at interpreter startup. A
regular __init__.py here would never execute and would shadow our submodules, so
the convenience exports live in this module instead. Import as
`from cuda.api import signal_ref, available, scff_signal`.
"""
from .reference import signal_ref


def available():
    import torch
    return torch.cuda.is_available()


def scff_signal(*a, **k):
    from .scff_ext import scff_signal as _f
    return _f(*a, **k)
