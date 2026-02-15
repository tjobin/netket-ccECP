import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf, fci, lo




class Molecule:
    """This class is to access the molecule and SCF calculations from pyscf."""
    def __init__(
            self,
            geometry,
            run_fci=True,
            verbose=True,
            charge=0,
            spin=0,
            basis='sto-3g',
            unit="Bohr"
    ):
        self.mol = gto.M(atom=geometry, basis=basis, unit=unit, charge=charge, spin=spin)
        self.mol.cart = True
        self.unit = unit
        assert unit.lower() == "bohr", "that's the correct one!"

        self._mf = None
        self._fci = None
        if verbose:
            print("Running restricted UHF (!)... (might want RHF)")
            print(self.mf.kernel())
        if verbose:
            print("UHF done.")
        self.ao_overlap = self.mf.mol.intor('int1e_ovlp_cart')
        if run_fci:
            if verbose:
                print("Running fci...")
                self.fci
            if verbose:
                print("fci done.")

    def info(self):
        print(self.mol.dump_input())

    @property
    def mf(self):
        "Hartree Fock energy"
        if self._mf is None:
            self._mf = scf.HF(self.mol).run()
        return self._mf

    @property
    def fci(self):
        """
        Full configuration interaction energy. This is the exact solution within the given basis set, and serves as a benchmark for our VMC calculations.
        Note that FCI can be very expensive for larger molecules or basis sets, so it may not always be feasible to run it.
        """
        if self._fci is None:
            self._fci = fci.FCI(self.mf).kernel()
        return self._fci

    def mo_boys(self, verbose=0):
        """
        Performs Foster-Boys localization on the molecular orbitals.

        Args:
            verbose (int): PySCF verbosity level. Defaults to 0 (quiet).

        Returns:
            numpy.ndarray: The localized orbital coefficients matrix (LMO 
                coefficients in the Atomic Orbital basis).
        """
        mo_boys = lo.Boys(self.mol).kernel(self.mf.mo_coeff, verbose=verbose)
        return mo_boys

    @property
    def n_basis(self):
        """
        Number of basis functions. This is the number of atomic orbitals (AOs) in the basis set, which determines the size of the MO
        coefficient matrix and the number of orbitals we have to work with.

        Returns:
            n_basis: list of int, number of basis functions for each atom in the molecule.
        """
        n_basis = [self.mol.bas_nprim(i) for i in range(self.mol.nbas)]
        return n_basis
        # return [self.mol.bas_atom(i) for i in range(self.mol.nbas)]

    @property
    def coordinates(self):
        """
        Returns the coordinates of the nuclei in the molecule.

        Returns:
            coords: numpy.ndarray of shape (N_n, 3) containing the coordinates of the nuclei.
        """
        coords = self.mol.atom_coords(unit=self.unit)
        return coords

    @property
    def n_orbitals(self):
        """
        Returns the number of molecular orbitals (MOs) obtained from the mean-field calculation.
        This is determined by the size of the MO coefficient matrix, which has dimensions (n_basis, n_orbitals).
        The number of MOs is typically equal to the number of basis functions, but can be different if there are
        symmetries or other constraints.

        Returns:
            n_orbitals: int, number of molecular orbitals.
        """
        n_orbitals = self.mf.mo_coeff.shape[0]
        return n_orbitals

    def atomic_orbitals(self):
        "Parameters etc. for the atomic orbitals (HF)."
        mol = self.mol
        assert mol.cart
        orbitals = []
        for i in range(mol.nbas):
            l = mol.bas_angular(i)
            i_atom = mol.bas_atom(i)
            zetas = mol.bas_exp(i)
            coeff_sets = mol.bas_ctr_coeff(i).T  # !!!
            # print(l, coeff_sets, _get_cartesian_angulars(l))
            for coeffs in coeff_sets:  # !!! this part I was missing
                # shells.append((i_atom, (l, coeffs, zetas)))
                # for lxyz in _get_cartesian_angulars(l):
                ao = {
                    "ind_nuc": i_atom,
                    "zetas": zetas,  # or alphas?
                    "weights": coeffs,
                    "ang_mom": l
                }
                orbitals.append(ao)
        # assert len(orbitals) == self.n_orbitals, f"got orbitals = {len(orbitals)} vs n_orbitals = {self.n_orbitals}"
        return orbitals

    @property
    def basis_set(self):
        return self.mol.basis

    @property
    def nuclear_charges(self):
        return self.mol.atom_charges()

    @property
    def n_electrons(self):
        return self.mol.tot_electrons()

    @property
    def n_per_spin(self):
        return self.mol.nelec

    @property
    def mf_mo_coefficients(self):
        "Coefficients from HF to combine atomic orbitals."
        # warnings.warn("taking mean-field mo_coeff")
        return self.mf.mo_coeff

    @property
    def EHF(self):
        return sum(self.mf.scf_summary.values())

    @property
    def EFCI(self):
        return self.fci[0]

    def __repr__(self):
        return f"Molecule(\n  {self.mol._atom},\n  basis={self.basis_set},\n  n_orbitals={self.n_orbitals},\n  n_electrons={self.n_per_spin}\n)"
    
def to_np_array(
        a: jnp.ndarray | np.ndarray | tuple | list | dict
        ) -> np.ndarray | tuple | list | dict:
    """
    Takes a JAX array, NumPy array, or a nested structure (tuple, list, dict) containing JAX or NumPy arrays
    and converts all arrays to NumPy arrays. This is useful for ensuring that the data is in a format compatible
    with PySCF and other libraries that expect NumPy arrays.

    Args:
        a: A JAX array, NumPy array, or a nested structure containing JAX or NumPy arrays.
    
    Returns:
        b: The same structure as `a`, but with all JAX arrays converted to NumPy arrays.
    """
    if isinstance(a, (np.ndarray, jnp.ndarray)):
        b = np.array(a) 
        return b
    elif isinstance(a, (tuple, list, dict)):
        b = jax.tree_util.tree_map(to_np_array, a)
        return b
    
def variables_from_mol(
        mol: Molecule
        ) -> tuple[np.ndarray, list[dict], list[np.ndarray], list[np.ndarray]]:
    """
    Returns the variables needed for the Slater-Jastrow wavefunction with backflow transformations.

    Args:
        mol: Molecule object containing the molecule and its properties.

    Returns:
        coords: np.ndarray of shape (N_n, d) containing the coordinates of the nuclei.
        atomic_orbitals: list of dicts containing the parameters for the atomic orbitals.
        mo_coeff_spin: list of np.ndarray of shape (n_orbitals, n_orbitals) containing the MO coefficients for each spin.
        ind_orb: list of np.ndarray containing the indices of the orbitals for each spin.
    """
    coords = to_np_array(mol.coordinates)  # (N_n, d)

    n_dets = 1
    n_up, n_dn = mol.n_per_spin

    ind_orb_up = np.arange(n_up).reshape(n_dets, -1)
    ind_orb_dn = np.arange(n_dn).reshape(n_dets, -1)
    ind_orb = [ind_orb_up, ind_orb_dn]

    mo_coeff = to_np_array(mol.mf_mo_coefficients)  # (n_orbitals, n_orbitals)

    # added from PauliNet
    mo_coeff = jnp.asarray(mo_coeff)
    ao_overlap = mol.ao_overlap
    mo_coeff *= jnp.sqrt(jnp.diag(ao_overlap))

    mo_coeff_spin = [mo_coeff[:,:n_up], mo_coeff[:,:n_dn]]  # they want it different per spin
    atomic_orbitals = mol.atomic_orbitals()

    return coords, atomic_orbitals, mo_coeff_spin, ind_orb
