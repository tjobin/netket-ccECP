import jax.numpy as jnp
import numpy as np
from utils import get_full_distance_matrix, get_el_ion_distance_matrix
from jax.scipy.special import sph_harm
from  ferminet_ecp.hamiltonian import non_local_energy, local_energy



REG_EPS = 10e-12

def get_el_el_potential_energy(r_el):
    "Coulomb interaction between electrons"
    assert r_el.ndim == 2
    n_el = r_el.shape[-2]
    eye = jnp.eye(n_el)
    dist_matrix = get_full_distance_matrix(r_el)
    
    # add eye to diagonal to prevent div/0
    E_pot = jnp.triu(1.0 / (dist_matrix + eye + REG_EPS), k=1)
    return jnp.sum(E_pot, axis=[-2, -1])

def get_ion_ion_potential_energy(R, Z):
    "Coulomb interaction between nucleons"
    assert R.ndim == 2
    n_ions = R.shape[-2]
    eye = jnp.eye(n_ions)
    dist_matrix = get_full_distance_matrix(R)
    charge_matrix = jnp.expand_dims(Z, -1) * jnp.expand_dims(Z, -2)

    # add eye to diagonal to prevent div/0
    E_pot = jnp.triu(charge_matrix / (dist_matrix + eye + REG_EPS), k=1)
    return jnp.sum(E_pot, axis=[-2, -1])

def get_potential_energy_ecp(r, model, mol, ecp_quadrature):
    "Total Coulomb interaction energy"
    E_pot_nonlocal = non_local_energy(r, model, mol, ecp_quadrature) # non-local ccECP
    E_pot_local= local_energy(r, mol.coordinates, mol.nuclear_charges, mol.n_electrons) # V_aa + V_ee + local ccECP
   
    return E_pot_nonlocal + E_pot_local
