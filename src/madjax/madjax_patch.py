import jax.numpy
from jax.numpy import sqrt


def max(a, b):
    return jax.numpy.max(jax.numpy.asarray([a, b]))


def min(a, b):
    return jax.numpy.min(jax.numpy.asarray([a, b]))


from jax.numpy import power as pow
from jax.numpy import pi
from itertools import product
import jax.numpy as np

assert sqrt
assert pow
assert pi
assert product


def complex(*v):
    if len(v) == 1:
        return np.asarray(v, dtype=np.complex64)
    else:
        return np.asarray(v[0] + 1j * v[1], dtype=np.complex64)
