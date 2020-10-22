from jax.numpy import sqrt
from jax.numpy import power as pow
from jax.numpy import pi
from itertools import product
import jax.numpy as jnp


def max(a, b):
    return jnp.max([a, b])


def min(a, b):
    return jnp.min([a, b])


def complex(*v):
    if len(v) == 1:
        return jnp.asarray(v, dtype=jnp.complex64)
    else:
        return jnp.asarray(v[0] + 1j * v[1], dtype=jnp.complex64)
