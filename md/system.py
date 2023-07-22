"""
This file contains a subclass of the schnetpack.system class.
In this state the simulation ignores pbc or cells.
"""
import torch

from typing import Union, List

from moleculekit.molecule import Molecule
from schnetpack.md import System
import schnetpack.units
from schnetpack.md.utils import NormalModeTransformer

from simulation_utils import AA2INT


class IPUSystem(System):
    def __init__(
            self, normal_mode_transform: NormalModeTransformer = NormalModeTransformer,
    ):
        #super(IPUSystem, self).__init__()
        torch.nn.Module.__init__(self)
        #super(IPUSystem, self).__init__(normal_mode_transform)
        self.properties = {}
        self._nm_transformer = normal_mode_transform
        # For initialized nm transform
        self.nm_transform = None

        # Dummy tensor for device and dtype
        self.register_buffer("_dd_dummy", torch.zeros(1))
        self.total_n_atoms = 0

    @property
    def device(self):
        return self._dd_dummy.device

    @property
    def dtype(self):
        return self._dd_dummy.dtype

    def load_molecules(self,
                       molecules: Union[Molecule, List[Molecule]],
                       n_replicas: int = 1,
                       position_unit_input: Union[str, float] = "Angstrom",
                       mass_unit_input: Union[str, float] = 1.0,
                       ):
        """
        This method fills the System with the initial inforation about the simulated molecule.
        The original SchNetPack.System loads ASE objects, which are not suitable for coarse graining.
        """
        if n_replicas > 1:
            raise NotImplementedError("Multiple replicas are not implemented.")

        self.n_replicas = n_replicas

        if isinstance(molecules, Molecule):
            molecules = [molecules]

        self.n_molecules = len(molecules)

        if self.n_molecules != 1:
            raise NotImplementedError("Multiple molecules are currently not tested.")

        # Register buffers
        self.register_buffer("n_atoms", torch.zeros(self.n_molecules, dtype=torch.int64))

        for i in range(self.n_molecules):
            self.n_atoms[i] = molecules[i].numAtoms

        self.total_n_atoms = torch.sum(self.n_atoms).item()

        # 3) Construct basic property arrays
        self.register_buffer("index_m", torch.zeros(self.total_n_atoms, dtype=torch.int64))
        self.register_buffer("atom_types", torch.zeros(self.total_n_atoms, dtype=torch.int64))
        self.register_buffer("masses", torch.zeros(1, self.total_n_atoms, 1))

        # Relevant for dynamic properties: positions, momenta, forces
        self.register_buffer("positions_buf", torch.zeros(self.n_replicas, self.total_n_atoms, 3))
        self.register_buffer("momenta_buf", torch.zeros(self.n_replicas, self.total_n_atoms, 3))
        self.register_buffer("forces_buf", torch.zeros(self.n_replicas, self.total_n_atoms, 3))

        self.register_buffer("energy_buf", torch.zeros(self.n_replicas, self.n_molecules, 1))

        mass2internal = schnetpack.units.unit2internal(1.0)  # out masses are in Dalton, which is standard
        positions2internal = schnetpack.units.unit2internal("Angstrom")

        offset = 0
        for mol_num, mol in enumerate(molecules):

            self.positions_buf[0][offset: offset + mol.numAtoms] = \
                torch.from_numpy(mol.coords.squeeze()) * positions2internal

            for index, aa_name in enumerate(mol.resname):
                self.atom_types[offset + index] = AA2INT[aa_name]
                self.masses[0, offset + index, 0] = mol.masses[index] * mass2internal
                self.index_m[offset + index] = mol_num
            else:
                offset += index + 1

        # TODO we do not need cells and pbc in out case so just leave it constant

        # Set normal mode transformer
        self.nm_transform = self._nm_transformer(n_replicas)

        # Properties for periodic boundary conditions and crystal cells
        self.register_buffer("cells", torch.zeros(self.n_replicas, self.n_molecules, 3, 3))
        self.register_buffer("pbc", torch.zeros(1, self.n_molecules, 3).bool())
        self.register_buffer(
            "stress", torch.zeros(self.n_replicas, self.n_molecules, 3, 3)
        )  # Used for the computation of the pressure

    @property
    def positions(self):
        return self.positions_buf

    @positions.setter
    def positions(self, new_positions):
        self.positions_buf.copy_(new_positions)

    @property
    def momenta(self):
        return self.momenta_buf

    @momenta.setter
    def momenta(self, new_momenta):
        self.momenta_buf.copy_(new_momenta)

    @property
    def forces(self):
        return self.forces_buf

    @forces.setter
    def forces(self, new_forces):
        self.forces_buf.copy_(new_forces)

    @property
    def energy(self):
        return self.energy_buf

    @forces.setter
    def energy(self, new_energy):
        self.energy_buf.copy_(new_energy)

    def expand_atoms(self, x: torch.Tensor):
        """
        Auxiliary routine for expanding molecular contributions over the corresponding atoms.

        Args:
            x (torch.Tensor): Tensor of the shape ( : x n_molecules x ...)

        Returns:
            torch.Tensor: Tensor of the shape ( : x (n_molecules * n_atoms) x ...)
        """
        return x[:, self.index_m, ...]

    def sum_atoms(self, x: torch.Tensor):
        """
        Auxiliary routine for summing atomic contributions for each molecule.

        Args:
            x (torch.Tensor): Input tensor of the shape ( : x (n_molecules * n_atoms) x ...)

        Returns:
            torch.Tensor: Aggregated tensor of the shape ( : x n_molecules x ...)
        """
        x_shape = x.shape
        x_tmp = torch.zeros(
            x_shape[0], self.n_molecules, *x_shape[2:], device=x.device, dtype=x.dtype
        )
        return x_tmp.index_add(1, self.index_m, x)

    @property
    def center_of_mass(self):
        """
        Compute the center of mass for each replica and molecule

        Returns:
            torch.Tensor: n_replicas x n_molecules x 3 tensor holding the
                          center of mass.
        """
        # Compute center of mass
        center_of_mass = self.sum_atoms(self.positions * self.masses) / self.sum_atoms(
            self.masses)
        return center_of_mass

    def remove_center_of_mass(self):
        """
        Move all structures to their respective center of mass.
        """
        self.positions -= self.expand_atoms(self.center_of_mass)

    @property
    def kinetic_energy(self) -> torch.tensor:
        """
        Convenience property for computing the kinetic energy associated with
        each replica and molecule.

        Returns:
            torch.Tensor: Tensor of the kinetic energies (in Hartree) with
                          the shape n_replicas x n_molecules x 1
        """
        kinetic_energy = 0.5 * self.sum_atoms(
            torch.sum(self.momenta**2, dim=2, keepdim=True) / self.masses
        )
        return kinetic_energy

    @property
    def temperature(self):
        """
        Convenience property for accessing the instantaneous temperatures of
        each replica and molecule.

        Returns:
            torch.Tensor: Tensor of the instantaneous temperatures (in
                          Kelvin) with the shape n_replicas x n_molecules x 1
        """
        temperature = (
            2.0
            / (3.0 * self.n_atoms[None, :, None] * schnetpack.units.kB)
            * self.kinetic_energy
        )
        return temperature

    def remove_translation(self):
        """
        Remove all components in the current momenta associated with
        translational motion.
        """
        self.momenta -= self.expand_atoms(self._mean_atoms(self.momenta))

    def _mean_atoms(self, x: torch.Tensor):
        """
        Auxiliary routine for computing mean over atomic contributions for each molecule.

        Args:
            x (torch.Tensor): Input tensor of the shape ( : x (n_molecules * n_atoms) x ...)

        Returns:
            torch.Tensor: Aggregated tensor of the shape ( : x n_molecules x ...)
        """
        return self.sum_atoms(x) / self.n_atoms[None, :, None]
