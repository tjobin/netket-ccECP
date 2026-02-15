import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import netket as nk
from netket.utils import struct
from netket.sampler import MetropolisSampler
from netket.hilbert import ContinuousHilbert
from .rules import AdaptiveGaussianRule, AdaptiveLangevinRule


@struct.dataclass
class MetropolisGaussAdaptive(MetropolisSampler):
    """Metropolis sampler that adaptively changes the sampler scale to get a given target acceptance."""

    target_acceptance: float = 0.6
    sigma_limits: Any = None

    def __pre_init__(self, *args, initial_sigma=1.0, sigma_limits=None, **kwargs):
        rule = AdaptiveGaussianRule(initial_sigma=initial_sigma)
        if sigma_limits is None:
            sigma_limits = [initial_sigma * 1e-2, initial_sigma * 1e2]
        kwargs["sigma_limits"] = tuple(sigma_limits)
        assert len(args) == 1, "should only pass hilbert"
        hilbert = args[0]
        args = [hilbert, rule]
        if not isinstance(hilbert, ContinuousHilbert):
            raise ValueError(
                f"This sampler only works for ContinuousHilbert Hilbert spaces, got {type(hilbert)}."
            )
        args, kwargs = super().__pre_init__(*args, **kwargs)
        return args, kwargs

    def _sample_next(self, machine, parameters, state):
        new_state, new_σ = super()._sample_next(machine, parameters, state)

        if self.target_acceptance is not None:
            acceptance = new_state.n_accepted / new_state.n_steps
            sigma = new_state.rule_state
            new_sigma = sigma / (
                self.target_acceptance
                / jnp.max(jnp.stack([acceptance, jnp.array(0.05)]))
            )
            new_sigma = jnp.max(jnp.array([new_sigma, self.sigma_limits[0]]))
            new_sigma = jnp.min(jnp.array([new_sigma, self.sigma_limits[1]]))
            new_rule_state = new_sigma
            new_state = new_state.replace(rule_state=new_rule_state)

        return new_state, new_σ


@struct.dataclass
class MetropolisLangevinAdaptive(MetropolisSampler):
    target_acceptance: float = None
    dt_limits: Any = None

    def __pre_init__(
        self, *args, initial_dt=1.0, dt_limits=None, chunk_size=None, **kwargs
    ):
        rule = AdaptiveLangevinRule(initial_dt=initial_dt, chunk_size=chunk_size)
        if dt_limits is None:
            dt_limits = [initial_dt * 1e-3, initial_dt * 1e3]
        kwargs["dt_limits"] = tuple(dt_limits)
        assert len(args) == 1, "should only pass hilbert"
        hilbert = args[0]
        args = [hilbert, rule]
        if not isinstance(hilbert, ContinuousHilbert):
            raise ValueError(
                f"This sampler only works for ContinuousHilbert Hilbert spaces, got {type(hilbert)}."
            )
        args, kwargs = super().__pre_init__(*args, **kwargs)
        return args, kwargs

    def _sample_next(self, machine, parameters, state):
        new_state, new_σ = super()._sample_next(machine, parameters, state)

        if self.target_acceptance is not None:
            acceptance = new_state.n_accepted / new_state.n_steps
            dt = new_state.rule_state
            new_dt = dt / (
                self.target_acceptance
                / jnp.max(jnp.stack([acceptance, jnp.array(0.05)]))
            )
            new_dt = jnp.max(jnp.array([new_dt, self.dt_limits[0]]))
            new_dt = jnp.min(jnp.array([new_dt, self.dt_limits[1]]))
            new_rule_state = new_dt
            new_state = new_state.replace(rule_state=new_rule_state)

        return new_state, new_σ
