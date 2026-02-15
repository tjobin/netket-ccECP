# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools

import jax
import jax.numpy as jnp

from ferminet_ecp.integral import special
from ferminet_ecp.integral.quadrature import Quadrature



def numerical_integral_exact(psi, r_atom, walkers, ls, quadrature: Quadrature):
    '''
    ref: Nonlocal pseudopotentials and diffusion Monte Carlo, equation 28

    inputs:
        psi: wave function psi(x) that returns a complex number
            with x shape (n_electron, 3)
        r_atom: shape (3,)
        walkers: shape (n_electron, 3)
        ls: shape(l_number,) values of l to evaluate
        quadrature: A quadrature object to do numerical integration.
    returns:
        value of the integral \int (2l+1) * P_l(cos theta) psi(r1,..,ri,..)
        shape (n_electron, l_number)
    '''

    n_electron = walkers.shape[0]
    ri = jnp.linalg.norm(walkers-r_atom, axis=-1)   # shape (n_electron,)
    res = jnp.zeros((n_electron, len(ls)))
    normal_walkers = (walkers-r_atom) / ri[:, None]
    psi_vec = jax.vmap(psi, in_axes=0) # psi_vec takes batch of samples (N, N_electrons, d) and returns (N,)

    for j, l in enumerate(ls):
        Pl_ = lambda x: special.legendre(x, l)

        def Pl(i, x):
            """
            Legendre polynomial of degree l evaluated at theta'. See equation (8).
            args:
                i: electron index, int
                x: integration points, shape (N_orientations, n_points, d)
            returns:
                res : wave function evaluated at each point, shape (N_orientations, n_points, 1)
            """
            tmp = Pl_(jnp.matmul(x, normal_walkers[i, :])) 
            return tmp

        def psi_r(i, x):
            """
            Returns psi(r1, ..., rv', ..., rN) - see equation (8) - where r_v is replaced by 
            points on the sphere of radius r_i and center r_atom. The default quadrature has
            N_orientations = 1 and n_points = 12.
            args:
                i: electron index, int
                x: integration points, shape (N_orientations, n_points, d)
            returns:
                res : wave function evaluated at (r1, ..., rv', ..., rN) at each point, shape (n_points, M)
            """
            coords = x.reshape(-1, 3) * ri[i] + r_atom
            new_walkers = jnp.tile(walkers, (coords.shape[0],) + (1, 1)) # (r1, ..., rv, ..., rN) of shape (n_points, n_electrons, d)
            new_walkers = new_walkers.at[:, i, :].set(coords)   # (r1, ..., rv', ..., rN) of shape (n_points, n_electrons, d)
            res = psi_vec(new_walkers)
            res = res.reshape(x.shape[:-1])
            return res

        def product(i, x):
            """
            args:
                i: electron index, int
                x: integration points, shape (M, n_points, n_electrons)
            returns:
                prod : integrand of the integral in eq (8), shape (M, n_points)
            """
            prod = Pl(i, x) * psi_r(i, x)
            return prod

        def integral(i, res):
            result = quadrature(lambda x: product(i, x)) * (2 * l + 1)
            res = res.at[i,j].set(result)       # MODIF : jax.ops.index_update is deprecated
            return res

        res = jax.lax.fori_loop(0, n_electron, integral, res)
    return res

