import jax
import jax.numpy as jnp
from flax import linen as nn

import numpy as np

from functools import partial
from typing import Sequence


from jax.config import config

config.update("jax_enable_x64", True)


def bisect_inverse(bijection, left_bound, right_bound, eps=1e-6):
    """Bisection search."""

    @jax.jit
    def _inverted(target):
        init = (jnp.ones_like(target) * left_bound, jnp.ones_like(target) * right_bound)
        n_iters = jnp.ceil(-jnp.log2(eps)).astype(int)

        def _body(_, val):
            left_bound, right_bound = val
            cand = (left_bound + right_bound) / 2
            pred = bijection(cand)
            left_bound = jnp.where(pred < target, cand, left_bound)
            right_bound = jnp.where(pred > target, cand, right_bound)
            return left_bound, right_bound

        return jax.lax.fori_loop(0, n_iters, _body, init)[0]

    return _inverted


def wrap_with_inverse(bijector, root_finder):
    def _forward(outp, cond):
        root = root_finder(lambda x: bijector(x, cond)[0])(outp)
        _, ldj = bijector(root, cond)
        return (root, -ldj), (root, cond)

    def _backward(res, tangents):
        root, cond = res
        root_grad, ldj_grad = tangents

        def _jac_diag(inp):
            outp, vjp_fun = jax.vjp(lambda x: bijector(x, cond)[0], inp)
            return vjp_fun(jnp.ones_like(outp))[0]

        jac_diag = _jac_diag(root)
        root_grad /= jac_diag
        ldj_grad /= jac_diag

        def _log_jac_diag(inp):
            return jnp.log(_jac_diag(inp))

        outp, pullback = jax.vjp(_log_jac_diag, root)
        dldj_dinp = pullback(jnp.ones_like(outp))[0]
        outp_grad = root_grad - dldj_dinp * ldj_grad

        # gradient wrt cond
        def _helper(cond):
            outp, _ = jax.vjp(lambda x: bijector(root, x)[0], cond)
            _, vjp_fun = jax.vjp(lambda x: bijector(x, cond)[0], root)
            jac = vjp_fun(jnp.ones_like(outp))[0]
            return (outp, outp, jac)

        _, pullback = jax.vjp(_helper, cond)
        cond_grad = pullback((-root_grad, dldj_dinp * ldj_grad, -ldj_grad))[0]
        return outp_grad, cond_grad

    @jax.custom_vjp
    def _call(outp, cond):
        return _forward(outp, cond)[0]

    _call.defvjp(_forward, _backward)

    return _call


@jax.jit
def tanh_bijector(x, cond, eps=1e-2):
    """Simple bijector with non-analytic inverse on [0,1]."""
    a, b = cond

    def _call(x):
        """Evaluate forward."""
        return 0.5 * (
            x[..., None] + (1.0 - eps) * jnp.tanh((x[..., None] - b) * a)
        ).sum(axis=-1)

    def _forward(x):
        """Evaluate and fit in [0,1]"""
        y0 = _call(jnp.zeros_like(x))
        y1 = _call(jnp.ones_like(x))
        return (_call(x) - y0) / (y1 - y0)

    y, vjp_fun = jax.vjp(_forward, x)
    _ldj = jnp.log(vjp_fun(jnp.ones_like(y))[0])
    ldj = _ldj.sum(axis=tuple(range(1, _ldj.ndim))).reshape(-1, 1)
    return y, ldj


@jax.jit
def tanh_bijector_inverse(x, cond):
    _inverse = wrap_with_inverse(
        tanh_bijector, partial(bisect_inverse, left_bound=0.0, right_bound=1.0)
    )
    return _inverse(x, cond)


@jax.jit
def smoothfun_bijector(x, cond, beta=2.0, eps=1e-6):
    """smooth bijector with non-analytic inverse from Smooth Normalizing Flows srXiv:2110.00351"""
    """a >= 1 (0?),   b in [0,1],   c in [0,1],  alpha > 0"""

    a, b, c, alpha = cond

    def _call(x):
        def rho(z):
            # _rho_val = jnp.where( z>0., jnp.exp(-1.0 / ((alpha+1.0)*jnp.power( jnp.where(z>0., z, 1.), beta))), 0.)
            _rho_val = jnp.where(
                z > 0.0, jnp.exp(-1.0 / ((alpha + 1.0) * jnp.power(z, beta))), 0.0
            )
            return _rho_val
            # _rho_val = jnp.exp(-1.0 / (alpha*jnp.power(z, beta)))
            # return jnp.where(z>0., _rho_val, 0.)

        def sigma(z):
            return rho(z) / (rho(z) + rho(1.0 - z))

        def g(z):
            return (1.0 - c) * sigma(a * (z - b) + 0.5) + (c + eps) * z

        return g(x[..., None]).sum(axis=-1)

    def _forward(x):
        """Evaluate and fit in [0,1]"""
        y0 = _call(jnp.zeros_like(x))
        y1 = _call(jnp.ones_like(x))
        return (_call(x) - y0) / (y1 - y0)

    y, vjp_fun = jax.vjp(_forward, x)
    _ldj = jnp.log(vjp_fun(jnp.ones_like(y))[0])
    ldj = _ldj.sum(axis=tuple(range(1, _ldj.ndim))).reshape(-1, 1)
    return y, ldj


@jax.jit
def smoothfun_bijector_inverse(x, cond):
    _inverse = wrap_with_inverse(
        smoothfun_bijector, partial(bisect_inverse, left_bound=0.0, right_bound=1.0)
    )
    return _inverse(x, cond)


class MLPconditioner(nn.Module):
    num_bij: int
    layer_widths: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        is_first = True
        for mlp in self.layer_widths:
            x = inputs
            for feat in mlp:
                x = nn.Dense(feat)(x)
                x = nn.swish(x)

            x = nn.Dense(4 * self.num_bij)(x)
            xa = jnp.exp(x[..., : self.num_bij])
            xb = nn.sigmoid(x[..., self.num_bij : 2 * self.num_bij])
            xc = nn.sigmoid(x[..., 2 * self.num_bij : 3 * self.num_bij])
            xalpha = jnp.exp(x[..., 3 * self.num_bij :])
            _out = jnp.expand_dims(jnp.hstack((xa, xb, xc, xalpha)), -2)

            if is_first:
                out = _out
                is_first = False
            else:
                out = jnp.concatenate((out, _out), axis=-2)

        return out


class SingleMLPconditioner(nn.Module):
    num_bij: int
    num_out_feat: int
    layer_widths: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for feat in self.layer_widths:
            x = nn.Dense(feat)(x)
            x = nn.swish(x)

        x = nn.Dense(4 * self.num_bij * self.num_out_feat)(x)
        x = jnp.reshape(x, x.shape[:-1] + (self.num_out_feat, 4 * self.num_bij))
        xa = jnp.exp(x[..., : self.num_bij])
        xb = nn.sigmoid(x[..., self.num_bij : 2 * self.num_bij])
        xc = nn.sigmoid(x[..., 2 * self.num_bij : 3 * self.num_bij])
        xalpha = jnp.exp(x[..., 3 * self.num_bij :])
        out = jnp.concatenate((xa, xb, xc, xalpha), axis=-1)

        return out


class SmoothFlowLayer(nn.Module):
    num_biject: int
    num_transform_feat: int
    cond_on_first: bool
    cond_mlp_width: Sequence[int]

    def setup(self):
        mlp_widths = self.num_transform_feat * self.cond_mlp_width
        self.conditioner = MLPconditioner(
            num_bij=self.num_biject, layer_widths=mlp_widths
        )

    def __call__(self, x):
        return self.inverse_bijection(x)

    def forward_bijection(self, z):
        if self.cond_on_first:
            z1 = z[..., 0 : -self.num_transform_feat]
            z2 = z[..., -self.num_transform_feat :]
        else:
            z1 = z[..., self.num_transform_feat :]
            z2 = z[..., 0 : self.num_transform_feat]

        K = self.num_biject

        _conds = self.conditioner(z1)

        a = _conds[..., 0:K]
        b = _conds[..., K : 2 * K]
        c = _conds[..., 2 * K : 3 * K]
        alpha = _conds[..., 3 * K :]

        cond = (a, b, c, alpha)

        y2, ldj = smoothfun_bijector(z2, cond)

        # y = jnp.where(self.cond_on_first, jnp.hstack((z1,y2)), jnp.hstack((y2,z1)))

        if self.cond_on_first:
            y = jnp.hstack((z1, y2))
        else:
            y = jnp.hstack((y2, z1))

        return y, ldj

    def inverse_bijection(self, y):
        if self.cond_on_first:
            y1 = y[..., 0 : -self.num_transform_feat]
            y2 = y[..., -self.num_transform_feat :]
        else:
            y1 = y[..., self.num_transform_feat :]
            y2 = y[..., 0 : self.num_transform_feat]

        K = self.num_biject

        _conds = self.conditioner(y1)

        a = _conds[..., 0:K]
        b = _conds[..., K : 2 * K]
        c = _conds[..., 2 * K : 3 * K]
        alpha = _conds[..., 3 * K :]

        cond = (a, b, c, alpha)

        z2, ldj = smoothfun_bijector_inverse(y2, cond)

        # z = jnp.where(self.cond_on_first, jnp.hstack((y1,z2)), jnp.hstack((z2,y1)))

        if self.cond_on_first:
            z = jnp.hstack((y1, z2))
        else:
            z = jnp.hstack((z2, y1))

        return z, ldj


class SmoothNormalizingFlow(nn.Module):
    num_flows: int
    num_biject: int
    num_in_feat: int
    cond_mlp_width: Sequence[int]

    def setup(self):
        self.flows = list(
            SmoothFlowLayer(
                num_biject=self.num_biject,
                num_transform_feat=self.num_in_feat // 2,
                cond_on_first=bool(i % 2),
                cond_mlp_width=self.cond_mlp_width,
            )
            for i in range(self.num_flows)
        )

    def __call__(self, x):
        return self.inverse_bijection(x)

    def forward_bijection(self, z):
        ldj = jnp.zeros(shape=(z.shape[0], 1))
        x = z

        for flow in self.flows:
            x, _ldj = flow.forward_bijection(x)
            ldj += _ldj

        return x, ldj

    def inverse_bijection(self, x):
        ldj = jnp.zeros(shape=(x.shape[0], 1))
        z = x

        for flow in reversed(self.flows):
            z, _ldj = flow.inverse_bijection(z)
            ldj += _ldj

        return z, ldj

    def logprob(self, x):
        z, ldj = self.inverse_bijection(x)
        base_logprob = (
            jax.scipy.stats.uniform.logpdf(z, loc=0, scale=1)
            .sum(axis=tuple(range(1, z.ndim)))
            .reshape(-1, 1)
        )
        return base_logprob + ldj

    def gradx_logprob(self, x):
        # def vgrad(f, x):
        #    y, vjp_fun = jax.vjp(f, x)
        #    return vjp_fun(jnp.ones(y.shape))[0]
        # return vgrad(lambda u: self.logprob(u), x)

        f = lambda x: self.logprob(x)
        y, vjp_fun = jax.vjp(f, x)
        return vjp_fun(jnp.ones(y.shape))[0]

    def val_and_gradx_logprob(self, x):
        f = lambda x: self.logprob(x)
        y, vjp_fun = jax.vjp(f, x)
        return y, vjp_fun(jnp.ones(y.shape))[0]

    def sample(self, key, N):
        z = jax.random.uniform(key, (N, self.num_in_feat))
        x, _ = self.forward_bijection(z)
        return x
