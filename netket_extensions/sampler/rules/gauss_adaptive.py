import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Optional, Any

from netket.utils import struct
import netket as nk
from netket.sampler import MetropolisRule


@struct.dataclass
class AdaptiveGaussianRule(MetropolisRule):
    r"""
    A transition rule acting on all particle positions at once.

    New proposals of particle positions are generated according to a
    Gaussian distribution of width sigma.
    """

    initial_sigma: float = 1.0
    """
    The variance of the gaussian distribution centered around the current
    configuration, used to propose new configurations.
    """

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params,
        key,
    ) -> Optional[Any]:
        sigma = self.initial_sigma
        return sigma

    def transition(rule, sampler, machine, parameters, state, key, r):
        if jnp.issubdtype(r.dtype, jnp.complexfloating):
            raise TypeError(
                "Adaptive Gaussian Rule does not work with complex " "basis elements."
            )

        n_chains = r.shape[0]
        hilb = sampler.hilbert

        pbc = np.array(hilb.n_particles * hilb.pbc, dtype=r.dtype)
        boundary = np.tile(pbc, (n_chains, 1))

        Ls = np.array(hilb.n_particles * hilb.extent, dtype=r.dtype)
        modulus = np.where(np.equal(pbc, False), jnp.inf, Ls)

        sigma = state.rule_state
        prop = jax.random.normal(
            key, shape=(n_chains, hilb.size), dtype=r.dtype
        ) * jnp.asarray(sigma, dtype=r.dtype)

        rp = jnp.where(np.equal(boundary, False), (r + prop), (r + prop) % modulus)

        return rp, None

    def __repr__(self):
        return f"AdaptiveGaussianRule(floating)"
