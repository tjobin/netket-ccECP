import jax
import jax.numpy as jnp
import numpy as np

geometries = {
    'LiH' : [('Li', (0,0,0)),
             ('H', (0,0,3.015))],
    'Li2' : [('Li', (0, 0, 0)),
             ('Li', (0, 0, 5.0512))],
    'Be2' : [('Be', (0,0,0)),
             ('Be', (0,0,4.6487))],
    'N2' : [('N', (0, 0, 1.0371)),
            ('N', (0, 0, -1.0371))],
    'NH3' : [('N', (0, 0, 0)),
             ('H', (0, -1.7720, -0.7211)),
             ('H', (1.5346, 0.8861, -0.7211)),
             ('H', (-1.5346, 0.8861, -0.7211))],
    'dimer' : [
            ('C',(15.2590006687, 16.8121532299, 15.9738065311)),
            ('O',(9.4486299386, 13.5060171431, 14.45823873)),
            ('H',(15.4105888182, 17.5070205934, 17.9029011803)),
            ('H',(15.2544898927, 18.4025938643, 14.6708461317)),
            ('H',(16.8591544947, 15.5899710576, 15.5578683925)),
            ('H',(13.5117713588, 15.749029294,  15.7636104197)),
            ('H',(7.68753004, 13.1203202896, 14.2934357266)),
            ('H',(10.167173679, 13.0333702175, 12.8652242888))
            ],
    'dimer_' : [
        ('C', (29.448623, 17.819358, 18.344269)),
        ('H', (30.915398, 19.144469, 18.909845)),
        ('H', (27.708179, 18.253679, 19.348981)),
        ('H', (29.128138, 17.975003, 16.319352)),
        ('H', (30.042775, 15.904281, 18.798897226)),
        ('O', (9.448630, 12.577577, 12.030276)),
        ('H', (10.973795, 12.307386, 11.093119)),
        ('H', (8.430457, 11.125273, 11.667596))
        ],
    'CH4' : [
        ('C', (29.448623, 17.819358, 18.344269)),
        ('H', (30.915398, 19.144469, 18.909845)),
        ('H', (27.708179, 18.253679, 19.348981)),
        ('H', (29.128138, 17.975003, 16.319352)),
        ('H', (30.042775, 15.904281, 18.798897226))
        ],
    'H2O' : [
        ('O', (9.4486299386, 12.5775758681, 12.0302753173)),
        ('H', (10.9737937682, 12.3073847361, 11.093118626)),
        ('H', (8.4304569148,  11.1252723137, 11.6675953263))
    ],
    'Ga' : [('Ga', (0, 0, 0))],
    'Kr' : [('Kr', (0, 0, 0))],
    'Sc' : [('Sc', (0, 0, 0))],
    'Li' : [('Li', (0, 0, 0))],
    'Be' : [('Be', (0, 0, 0))],
    'B' : [('B', (0, 0, 0))],
    'C' : [('C', (0, 0, 0))],
    'H' : [('H', (0, 0, 0))],
    }

def get_el_ion_distance_matrix(
        r_el: jnp.ndarray, # shape [N_batch x n_el x 3]
        R_ion: jnp.ndarray  # shape [N_ion x 3]
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes distance vectors and their norm between inputs
    Args:
        r_el: shape [N_batch x n_el x 3]
        R_ion: shape [N_ion x 3]
    Returns:
        diff: shape [N_batch x n_el x N_ion x 3]
        dist: shape [N_batch x n_el x N_ion]
    """
    diff = r_el[..., None, :] - R_ion[..., None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist

def get_full_distance_matrix(
        r_el: jnp.ndarray
        ) -> jnp.ndarray:
    """
    Computes distance vectors between inputs
    Args:
        r_el: jnp.array of shape [n_el x 3], contains 
    Returns:
        dist: shape [n_el x n_el], distances between electrons
    """
    diff = jnp.expand_dims(r_el, -2) - jnp.expand_dims(r_el, -3)
    dist = jnp.linalg.norm(diff, axis=-1)
    return dist

def dists_from_diffs_matrix(
        r_el_diff: jnp.ndarray
        ) -> jnp.ndarray:
    n_el = r_el_diff.shape[-2]
    diff_padded = r_el_diff + jnp.eye(n_el)[..., None]
    dist = jnp.linalg.norm(diff_padded, axis=-1) * (1 - jnp.eye(n_el))
    return dist

def get_distance_matrix(
        r_el: jnp.ndarray
        ) -> tuple[jnp.array, jnp.array]: #  stable!
    """
    Compute distance matrix omitting the main diagonal (i.e. distance to the particle itself)
    Args:
        r_el: [batch_dims x n_electrons x 3]
    Returns:
        diff: jnp.array of shape [batch_dims, n_el, n_el, 3], 
        dist: jnp.array of shape [batch_dims, n_el, n_el], distances
    """
    diff = r_el[..., :, None, :] - r_el[..., None, :, :]
    dist = dists_from_diffs_matrix(diff)
    return diff, dist

REG_EPS = 1e-12


def v(r, Rn, Zn):
    """
    Calculates the Coulomb potential energy in H2

    Args:
        r: jnp.array of shape (N*d,), contains the xyz coordinates of the 2 electrons
        Rn: jnp.array of shape (Nn*d,), contains the xyz coordinates of the two nuclei
        Zn: jnp.array of shape (Nn,), contains the electric charges of the two nuclei
    Returns:
        e_pot: jnp.float, Coulomb potential energy 
    """


    dist_ee = get_distance_matrix(r)             # jnp.array of size (1, N*(N-1)/2), first axis due to calculate_dist
    dist_nn = get_distance_matrix(Rn)            # jnp.array of size (1, Nn*(Nn-1)/2)
    dist_en = get_el_ion_distance_matrix(r, Rn)         # jnp.array of size (1, N, Nn)
    Nn = len(Rn) // 3 

    Znn = jnp.einsum("i,j->ij", Zn, Zn)[jnp.triu_indices(Nn, k=1)]     # (Nn*(Nn-1)/2,), products of nucleus charges

    # calculate the three Coulomb potential energy terms
    arg_pot_ee = 1 / (dist_ee + REG_EPS)                                #  (N*(N-1)/2,)
    pot_ee = jnp.sum(arg_pot_ee)

    arg_pot_nn = jnp.einsum('i,...i->...i', Znn, 1/(dist_nn + REG_EPS))       # (Nn*(Nn-1)/2,), (1,Nn*(Nn-1)/2) -> (1,Nn*(Nn-1)/2)
    pot_nn = jnp.sum(arg_pot_nn)

    arg_pot_en = jnp.einsum('j,...ij->...ij', Zn, 1/(dist_en + REG_EPS))      # (Nn,), (N,Nn) -> (N, Nn)
    pot_en = -jnp.sum(arg_pot_en)

    e_pot = pot_ee + pot_nn + pot_en
    
    return e_pot

def create_Phi(dist_en_sigma, pi_sigma, c_sigma):
    """
    Creates an array Phi_up or Phi_down

    inputs:
        dist_rR_sigma : jnp.array of shape (Ns, N_sigma, Nn)
        pi_sigma : jnp.array of shape (N_states_sigma,), decay parameters 
        c_sigma : jnp.array of shape (N_states_sigma,), weight parameters
    returns:
        Phi_sigma : jnp.array of shape (Ns, N_states_sigma, N_sigma)
    """
    exp_arg_sigma = -jnp.einsum("i,...jk->...ijk", pi_sigma, dist_en_sigma) 
    exp_sigma = jnp.einsum("i, ...ijk->...ijk", c_sigma, jnp.exp(exp_arg_sigma))
    Phi_sigma = jnp.sum(exp_sigma, axis = -1)

    return Phi_sigma

def pseudopotential(r, params):
    """
    Returns a pseudo-potential of the form ar / (1 + br)

    inputs:
        r : jnp.array of size (Ns, N_dist), contains all interparticle distances of interest
        params : jnp.array of size (2, 1), contains the jastrow params a and b
    returns:
        u : jnp.array of size (Ns, N_dist), contains the pseudo-potentials
    """

    a, b = params[0], params[1]
    u = a*r / (1 + b*r)

    return u

