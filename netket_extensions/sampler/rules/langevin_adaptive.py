import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import struct
import netket as nk
from functools import partial

from typing import Optional, Any
from netket.sampler import MetropolisSampler, MetropolisRule
import netket.jax as nkjax

from netket.hilbert import ContinuousHilbert

import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class AdaptiveLangevinRule(MetropolisRule):
    r"""
    A transition rule that uses Langevin dynamics [1] to update samples.

    .. math::
       x_{t+dt} = x_t + dt \nabla_x \log p(x) \vert_{x=x_t} + \sqrt{2 dt}\eta,

    where  :math:`\eta` is normal distributed noise :math:`\eta \sim \mathcal{N}(0,1)`.
    This rule only works for continuous Hilbert spaces.

    [1]: https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm
    """

    initial_dt: float = 0.001
    """
    Time step in the Langevin dynamics
    """
    chunk_size: int = struct.field(pytree_node=False, default=None)
    """
    Chunk size for computing gradients of the ansatz
    """

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params,
        key,
    ) -> Optional[Any]:
        dt = self.initial_dt
        return dt

    def transition(rule, sampler, machine, parameters, state, key, r):
        if jnp.issubdtype(r.dtype, jnp.complexfloating):
            raise TypeError("LangevinRule does not work with complex basis elements.")

        n_chains = r.shape[0]
        hilb = sampler.hilbert

        pbc = np.array(hilb.n_particles * hilb.pbc, dtype=r.dtype)
        boundary = np.tile(pbc, (n_chains, 1))

        Ls = np.array(hilb.n_particles * hilb.extent, dtype=r.dtype)
        modulus = np.where(np.equal(pbc, False), jnp.inf, Ls)

        # one langevin step
        dt = state.rule_state
        rp, log_corr = _langevin_step(
            key,
            r,
            machine.apply,
            parameters,
            sampler.machine_pow,
            dt,
            chunk_size=rule.chunk_size,
            return_log_corr=True,
        )

        rp = jnp.where(np.equal(boundary, False), rp, rp % modulus)

        return rp, log_corr

    def __repr__(self):
        return f"LangevinRule()"


@partial(jax.jit, static_argnames=("apply_fun", "chunk_size", "return_log_corr"))
def _langevin_step(
    key,
    r,
    apply_fun,
    parameters,
    machine_pow,
    dt,
    chunk_size=None,
    return_log_corr=True,
):
    """Single step of samples with Langevin dynamics"""

    n_chains, hilb_size = r.shape

    # steps with Langevin dynamics
    noise_vec = jax.random.normal(key, shape=(n_chains, hilb_size), dtype=r.dtype)

    def _log_prob(x):
        """Conversion to a log probability"""
        return machine_pow * apply_fun(parameters, x).real

    def _single_grad(x):
        """Derivative of log_prob with respect to a single sample x"""
        x = x.reshape(x.shape[-1])
        g = nkjax.grad(lambda xi: _log_prob(xi).ravel()[0])(x)
        return g if jnp.iscomplexobj(r) else g.real

    grad_logp_r = nkjax.vmap_chunked(_single_grad, chunk_size=chunk_size)(r)

    rp = r + dt * grad_logp_r + jnp.sqrt(2 * dt) * noise_vec

    if not return_log_corr:
        return rp
    else:
        log_q_xp = -0.5 * jnp.sum(noise_vec**2, axis=-1)
        grad_logp_rp = nkjax.vmap_chunked(_single_grad, chunk_size=chunk_size)(rp)
        log_q_x = -jnp.sum((r - rp - dt * grad_logp_rp) ** 2, axis=-1) / (4 * dt)

        return rp, log_q_x - log_q_xp
