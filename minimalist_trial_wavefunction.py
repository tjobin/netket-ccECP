from jax import numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros, ones
from netket.jax._math import logdet_cmplx
from utils import get_distance_matrix, get_el_ion_distance_matrix, v, pseudopotential, create_Phi
from pyscf_molecule import Molecule

def LogSlaterDet(
        r: jnp.array,
        Rn: jnp.array,
        N_up: int,
        N_down: int,
        slater_params
        ):
    """
    Calculate the Slater determinant for the two electrons in H2

    inputs:
        r: jnp.array of size (Ns, N*d), contains the x, y, z coordinates
           of the electrons
        slater_params: jnp.array of size (8,1), contains the variational 
                       parameters of each electron for the slater part
    returns:
        log_slater_det: jnp.array of size (Ns,), contains the Slater determinant of each sample
    """
    
    _, dist_en = get_el_ion_distance_matrix(r, Rn)        # jnp.array of size (Ns, N=4, Nn=2)
    dist_en_up = dist_en[:, :N_up, :]
    dist_en_down = dist_en[:, -N_down:, :]
    pi_up, c_up = slater_params[:N_up], slater_params[N_up:2*N_up]             # shape (N_states_up,) and (N_states_down)
    pi_down, c_down = slater_params[-2*N_down:-N_up], slater_params[-N_down:]  # shape (N_states_up,), and (N_states_down)


    Phi_up = create_Phi(dist_en_up, pi_up, c_up)                # shape (Ns, N_states_up, N_up)
    Phi_down = create_Phi(dist_en_down, pi_down, c_down)        # shape (Ns, N_states_down, N_down)

    logdet_up = logdet_cmplx(Phi_up)
    logdet_down = logdet_cmplx(Phi_down)

    log_slater_det = logdet_up + logdet_down
    
    return log_slater_det


def JastrowFactor(
        r,
        Rn,
        jastrow_params
        ):
    """
    Calculate the Jastrow factor (pseudopotential) for the two electrons in H2

    Args:
        r: jnp.array of size (Ns, N*d), contains the x, y, z coordinates
           of the electrons
        jastrow_params: jnp.array of size (2,), contains the variational 
                       parameters of each electron for the jastrow part

    Returns:
        j_factor: jnp.array of size (Ns,), contains the jastrow factor of each sample
    """
    
    _, dist_ee = get_distance_matrix(r)                             # (Ns, N*(N-1)/2)
    _, dist_en = get_el_ion_distance_matrix(r, Rn)                        # jnp.array of size (Ns, N, Nn)
    dist_ee = jnp.sum(jnp.triu(dist_ee, k=1), axis=-1)
    dist_en = jnp.reshape(dist_en, (jnp.shape(dist_en)[0], -1))     # flattens the 3D array (Ns, N, Nn) into a 2D array (Ns, N*Nn)

    jastrow_params_ee = jastrow_params[0:2]
    jastrow_params_en = jastrow_params[2:4]
    j_ee = jnp.sum(pseudopotential(dist_ee, jastrow_params_ee), axis = -1)      
    j_en = - jnp.sum(pseudopotential(dist_en, jastrow_params_en), axis = -1)
    j_factor = j_ee + j_en

    return j_factor

class Minimalist(nn.Module):
    mol: Molecule

    def setup(self):
        self.Rn = self.mol.coordinates
        self.N_up, self.N_down = self.mol.n_per_spin

    @nn.compact
    def __call__(self, r):

        def logpsi(r):
            r = r.reshape(r.shape[0], -1, 3)
            # Create the Slater Determinant
            slater_params = self.param('slater_params', ones, (8,), float)
            log_slater_det = jnp.real(LogSlaterDet(
                r,
                self.Rn,
                self.N_up,
                self.N_down,
                slater_params
                ))

            # Construct the Jastrow factor
            jastrow_params = self.param('jastrow_params', ones, (4,), float)
            Jastrow = JastrowFactor(
                r,
                self.Rn,
                jastrow_params
                )

            return log_slater_det + Jastrow
            
        return logpsi(r)