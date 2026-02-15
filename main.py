import sys
import jax.numpy as jnp
from pyscf_molecule import Molecule
import netket as nk
import netket_extensions as nkext
from optax._src import linear_algebra
from _potential import PotentialEnergy
from potential_ecp import get_potential_energy_ecp
from potential import get_potential_energy
from utils import make_ecp, geometries
from minimalist_trial_wavefunction import Minimalist
import flax

## Import optimized params from another run
load_params = False

## Mol object
molecule = 'Ga'
geometry = geometries(molecule)
basis_set =  'sto-3g' #'ccecpccpvdz'
charge = 0
spin = 0
ecp = make_ecp(geometry)

mol = Molecule(geometry, ecp=ecp, run_fci=False, basis=basis_set, unit='Bohr', charge=charge, spin=spin)

## Ansatz parameters
global_feature = False
N_orbitals=8
intermediate_dim=8
mlp_output_dim=intermediate_dim
mlp_layers=2
normalization=True
attention_dim=intermediate_dim
n_features=8
n_interactions=1
n_heads=2

## Sampler parameters
n_chains_per_rank = 8
n_samples = 1024
sweep_size = 32
n_discard_per_chain = 32
chunk_size = None

## Optimizer parameters
opt_name = 'Sgd'
lr = 0.01
diag_shift = 0.001

## Output filename
sctach_path = f''
filename_head = f'{molecule}_MinimalistAnsatz_ecp'
logfile_name = sctach_path + f'data_log/' + filename_head

hilb = nk.hilbert.Particle(N=mol.n_electrons, L=(jnp.inf,jnp.inf,jnp.inf), pbc=False)
sampler = nkext.sampler.MetropolisGaussAdaptive(
    hilb,
    initial_sigma=0.05,
    target_acceptance=0.5,
    n_chains_per_rank=n_chains_per_rank,
    sweep_size=sweep_size
)

model = Minimalist(
        mol=mol,
        )


potential = lambda x, model: get_potential_energy_ecp(x, model, mol, ecp_quadrature='icosahedron_12')

epot = PotentialEnergy(hilb, potential)
ekin = nk.operator.KineticEnergy(hilb, mass=1.)
ham = ekin + epot
vs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain, chunk_size=None)

if load_params:
    mpack_filename = sctach_path + f'data_log/' + filename_head + '.mpack'
    with open(mpack_filename, 'rb') as file:
        vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
        print("optimized params imported")

## Define optimizer and 
op = nk.optimizer.Sgd(lr)
sr = nk.optimizer.SR(diag_shift=diag_shift)

def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    logged_data["globalnorm"] = float(linear_algebra.global_norm(driver._loss_grad))
    return True

log = nk.logging.JsonLog(logfile_name, save_params_every=25)
gs = nk.VMC(ham, op, variational_state=vs, preconditioner=sr)

gs.run(n_iter=20000, callback=mycb, out=log)

