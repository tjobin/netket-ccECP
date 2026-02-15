# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2021 Bytedance Inc.

"""Evaluating the Hamiltonian on a wavefunction."""

from utils import get_el_ion_distance_matrix, get_full_distance_matrix
import jax
from jax import lax
import jax.numpy as jnp

from ferminet_ecp.integral import pseudoPotential
from ferminet_ecp.integral.quadrature import get_quadrature

def potential_energy(r_ae, r_ee, atoms, charges):
    """Returns the potential energy for this electron configuration.

    Args:
      r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
        electron i and atom j.
      r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
        between electrons i and j. Other elements in the final axes are not
        required.
      atoms: Shape (natoms, ndim). Positions of the atoms.
      charges: Shape (natoms). Nuclear charges of the atoms.

    MODIFICATION FROM FERMINET: None
    """
    v_ee = jnp.sum(jnp.triu(1 / r_ee, k=1))
    v_ae = -jnp.sum(charges / r_ae)  # pylint: disable=invalid-unary-operand-type
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    v_aa = jnp.sum(
        jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
    return v_ee + v_ae + v_aa


def local_energy(x, atoms, charges, N):
    """Creates function to evaluate the local energy.

    Args:
      x: Shape (N * ndim). MCMC configuration.
      atoms: Shape (natoms, ndim). Positions of the atoms.
      charges: Shape (natoms). Nuclear charges of the atoms.
      N: Number of electrons.

    Returns:
      potential : the three Coulombic potentials 
        vee = \sum i>j (1 / r_ij)
        vae = \sum i,I (Zeff_I / r_iI)
        vaa = \sum I>J (Zeff_I * Zeff_J / r_IJ)
    """

    r_ae = jnp.reshape(get_el_ion_distance_matrix(x.reshape((-1, N, 3)), atoms)[1], (-1, atoms.shape[0]))
    r_ee = get_full_distance_matrix(x.reshape(N, 3))
    
    potential = potential_energy(r_ae, r_ee, atoms, charges)
    return potential



def ecp(pe, pa, ecp_coe):
    """
    read ecp coeffs from pyscf obj

    NEWLY ADDED
    """
    norm = jnp.linalg.norm(pe[:, None, :] - pa, axis=-1)
    res = []
    for l, coeff_ls in ecp_coe:
        result = 0
        for power, coe in enumerate(coeff_ls):
            for coeff in coe:
                result = result + norm[:, 0] ** (power - 2) * jnp.exp(- coeff[0] * norm[:, 0] ** 2) * \
                         coeff[1]
        res.append(result) # result has shape (N,)
    res = jnp.stack(res, axis=-1) # each column is the result for each l, each row 
    return res # shape (N, l_max + 1)


def non_local_energy(x, fs, mole, ecp_quadrature_id=None):
    """
    Calculate Ecp energy.
    Args:
        x : (N*d,) array
            The electron positions.
        fs : function with signature fs(x)
            The wavefunction, with x (..., N*d) array --> (..., 1) array
        mole : Molecule object
            The molecule object.
    Returns:
        res : float
            The ccECP energy Vloc + \sum_l V_l |lm><lm|
    """
    quadrature = get_quadrature(ecp_quadrature_id)
    N = mole.n_electrons
    d = 3

    def psi(x):
        """x : (N, d) array"""
        logpsi = fs(x.reshape(-1, N*d))

        return jnp.exp(logpsi)

    def non_local(pe, pa, psi, l_list):
        """
        pe : (N, d) array
            The electron positions.
        pa : (d,) array
            The atom position.
        psi : function with signature psi(x) and x is a (N, d) array
            The wavefunction.
        l_list : list of length l_max + 1 [0, ..., l_max] (l_max is the highest angular momentum in the ECP)"""
        res = pseudoPotential.numerical_integral_exact(psi, pa, pe, l_list, quadrature)
        return res / (4 * jnp.pi * psi(pe))

    def non_local_sum(x):
        res = 0
        pe = x.reshape(N, d)
        for sym, coord in mole.mol._atom:
            result = 0
            if sym in mole.mol._ecp:
                pa = jnp.array(coord)
                ecp_coe = mole.mol._ecp[sym][1]
                l_list = list(range(len(ecp_coe) - 1))
                ecp_list = ecp(pe, pa, ecp_coe) # shape (N, l_max + 1)
                result = (
                    jnp.sum(ecp_list[..., 1:] * non_local(pe, pa, lambda x: psi(x.flatten()), l_list),
                                  axis=-1) # sum over angular momenta l
                        + ecp_list[..., 0]
                        ) # sum over electrons 
            res = res + result
        return jnp.sum(res, axis=-1) # sum over electrons

    return non_local_sum(x)