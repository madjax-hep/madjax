import jax.numpy
from jax.numpy import sqrt


def max(a, b):
    return jax.numpy.max([a, b])


def min(a, b):
    return jax.numpy.min([a, b])


from jax.numpy import power as pow
from jax.numpy import pi
from itertools import product
import jax.numpy as np


def complex(*v):
    if len(v) == 1:
        return np.asarray(v, dtype=np.complex64)
    else:
        return np.asarray(v[0] + 1j * v[1], dtype=np.complex64)
